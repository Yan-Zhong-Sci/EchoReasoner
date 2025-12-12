# -*- coding: utf-8 -*-
"""
ECHOReasoner：Qwen-VL + 文本先验聚焦 + Mask2Former 分割 + 掩膜证据回灌 + 槽位推理。
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen_vl_base import QwenVLBase
from .text_guided_cropper import TextGuidedCropper
from .pixel_decoder import Mask2FormerSegmentation
from .qwen_pixel_bridge import QwenPixelBridge
from .slot_reasoner import SlotReasoner


class ECHOReasoner(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        """
        cfg 示例（与 README 约定一致）：
          cfg["base_model"] = {
            "name": "Qwen2.5-VL-3B-Instruct",
            "vision_hidden_size": 1024,
            "text_hidden_size": 2048,
            "vision_patch_size": 14,
            "special_tokens": ["<SEG#1>", "<EVIDENCE>", "</EVIDENCE>"],
          }
          cfg["text_guided_cropper"] = { ... }
          cfg["pixel_decoder"] = { "hidden_dim":256, "num_queries":100, ... }
          cfg["bridge"] = { "seg_token": "<SEG#1>", "max_masks":5 }
          cfg["slot_reasoner"] = { "max_answer_len":128 }
        """
        super().__init__()

        base_cfg = cfg.get("base_model", {})
        model_name = base_cfg.get("name")
        if model_name is None:
            raise KeyError("cfg['base_model']['name'] 不能为空。")

        special_tokens = base_cfg.get("special_tokens", [])
        self.qwen = QwenVLBase(
            model_name=model_name,
            vision_hidden_size=base_cfg.get("vision_hidden_size", 1024),
            text_hidden_size=base_cfg.get("text_hidden_size", 2048),
            vision_patch_size=base_cfg.get("vision_patch_size", 14),
            special_tokens=special_tokens,
            load_lora=base_cfg.get("load_lora", False),
            lora_r=base_cfg.get("lora_r", 8),
            lora_alpha=base_cfg.get("lora_alpha", 16),
            lora_dropout=base_cfg.get("lora_dropout", 0.05),
            lora_target_modules=base_cfg.get("lora_target_modules", None),
            lora_layer_type=base_cfg.get("lora_layer_type", "qkvo"),
            lora_modules_scope=base_cfg.get("lora_modules_scope", "all"),
            lora_layer_type_llm=base_cfg.get("lora_layer_type_llm", base_cfg.get("lora_layer_type", "qkvo")),
            lora_layer_type_visual=base_cfg.get("lora_layer_type_visual", base_cfg.get("lora_layer_type", "qkvo")),
            lora_target_modules_llm=base_cfg.get("lora_target_modules_llm", None),
            lora_target_modules_visual=base_cfg.get("lora_target_modules_visual", None),
            lora_include_llm=base_cfg.get("lora_include_llm", True),
            lora_include_visual=base_cfg.get("lora_include_visual", True),
            lora_r_llm=base_cfg.get("lora_r_llm", None),
            lora_r_visual=base_cfg.get("lora_r_visual", None),
            lora_alpha_llm=base_cfg.get("lora_alpha_llm", None),
            lora_alpha_visual=base_cfg.get("lora_alpha_visual", None),
            lora_dropout_llm=base_cfg.get("lora_dropout_llm", None),
            lora_dropout_visual=base_cfg.get("lora_dropout_visual", None),
            vision_layers=base_cfg.get("vision_layers", None),
        )

        crop_cfg = cfg.get("text_guided_cropper", {})
        focus_token = crop_cfg.get("focus_token", None)
        focus_token_id = None
        if isinstance(focus_token, str):
            focus_token_id = self.qwen.get_special_token_id(focus_token)
        elif focus_token is not None:
            focus_token_id = int(focus_token)

        self.cropper = TextGuidedCropper(
            vision_dim=self.qwen.vision_hidden_size,
            text_dim=self.qwen.text_hidden_size,
            hidden_dim=crop_cfg.get("hidden_dim", 256),
            focus_token_id=focus_token_id,
            topk_ratio=crop_cfg.get("topk_ratio", 0.2),
            background_scale=crop_cfg.get("background_scale", 0.5),
        )

        pd_cfg = cfg.get("pixel_decoder", {})
        hidden_dim = pd_cfg.get("hidden_dim", 256)
        num_queries = pd_cfg.get("num_queries", 100)
        self.segmentation = Mask2FormerSegmentation(
            vision_channels=self.qwen.vision_hidden_size,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=pd_cfg.get("num_layers", 6),
            num_heads=pd_cfg.get("num_heads", 8),
            num_points=pd_cfg.get("num_points", 4),
            dropout=pd_cfg.get("dropout", 0.1),
            pretrained_path=pd_cfg.get("pretrained_path", None),
            pretrained_strict=pd_cfg.get("pretrained_strict", False),
            freeze=pd_cfg.get("freeze", False),
        )

        bridge_cfg = cfg.get("bridge", {})
        seg_token = bridge_cfg.get("seg_token", "<SEG#1>")
        seg_token_id = (
            self.qwen.get_special_token_id(seg_token)
            if isinstance(seg_token, str)
            else int(seg_token)
        )
        self.bridge = QwenPixelBridge(
            text_hidden_dim=self.qwen.text_hidden_size,
            query_dim=hidden_dim,
            seg_token_id=seg_token_id,
            evidence_token_dim=hidden_dim,
            max_masks=bridge_cfg.get("max_masks", 5),
        )
        self.evidence_token_id = (
            self.qwen.get_special_token_id(bridge_cfg.get("evidence_token", "<EVIDENCE>"))
            if isinstance(bridge_cfg.get("evidence_token", None), str)
            else bridge_cfg.get("evidence_token", None)
        )

        sr_cfg = cfg.get("slot_reasoner", {})
        self.slot_reasoner = SlotReasoner(
            llm=self.qwen.model,
            tokenizer=self.qwen.tokenizer,
            max_answer_len=sr_cfg.get("max_answer_len", 128),
        )

        self.num_queries = num_queries

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "infer",
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        参数：
          batch: {
            "pixel_values": [B,3,H,W],
            "input_ids": [B,L],
            "attention_mask": [B,L],
            "labels": [B,L] (可选，SFT 用)
          }
          mode: 'sft' | 'grpo' | 'infer'
        返回：
          包含分割输出、证据、生成文本等字段的 dict。
        """
        images = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None) if mode == "sft" else None

        # 1) Qwen 编码图文
        qwen_out = self.qwen(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_hidden = qwen_out["text_hidden"]
        text_hidden_all = qwen_out.get("text_hidden_all", None)
        vision_map = qwen_out["vision_map"]
        vision_maps = qwen_out.get("vision_maps", None)
        attn_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

        # 2) 文本先验聚焦
        crop_out = self.cropper(
            vision_map=vision_map,
            text_hidden=text_hidden_all if text_hidden_all is not None else text_hidden,
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        focus_map = crop_out["focus_map"]
        focused_feat = crop_out["focused_feature"]
        focused_feats = None
        if vision_maps is not None:
            focused_feats = []
            for vm in vision_maps:
                fm = F.interpolate(focus_map, size=vm.shape[-2:], mode="bilinear", align_corners=False)
                focused_vm = vm * (fm + self.cropper.background_scale * (1 - fm))
                focused_feats.append(focused_vm)

        # 3) 像素分割
        feats_for_seg = focused_feats if focused_feats is not None else [focused_feat]
        # 3) 像素分割
        base_queries = self.segmentation.query_embed.weight  # [K,Dq]
        query_init, query_gate = self.bridge.language_to_queries(
            text_hidden=text_hidden,
            input_ids=input_ids,
            base_queries=base_queries,
        )
        # 构造与分割输入同尺度的 bias（预先生成一次金字塔用于 bias 计算）
        bias_pyramid = self.segmentation._fpn_build(feats_for_seg if isinstance(feats_for_seg, list) else [feats_for_seg])
        attn_bias = self.bridge.make_attn_bias(
            focus_map=focus_map,
            pixel_feats=bias_pyramid,
            num_queries=self.num_queries,
        )
        seg_out = self.segmentation(
            features=feats_for_seg,
            query_init=query_init,
            attn_bias=attn_bias,
            query_gate=query_gate,
        )

        # 4) 掩膜 -> 证据 token
        evidence = self.bridge.masks_to_evidence(
            mask_logits=seg_out["logits"],
            pixel_feat=seg_out["pixel_feats"][8],
            mask_scores=seg_out["scores"],
        )  # [B,M,Dt]

        # 5) 构造嵌入并插入 evidence
        inputs_embeds = self.qwen.embed_tokens(input_ids)
        evidence_token_id = (
            self.evidence_token_id
            if self.evidence_token_id is not None
            else int(self.qwen.get_special_token_id("<EVIDENCE>") or self.qwen.tokenizer.eos_token_id)
        )
        pad_id = getattr(self.qwen.tokenizer, "pad_token_id", 0)
        injected = self.bridge.inject_evidence_tokens(
            input_ids=input_ids,
            attention_mask=attn_mask,
            inputs_embeds=inputs_embeds,
            evidence_embeds=evidence,
            evidence_token_id=evidence_token_id,
            pad_token_id=pad_id,
        )

        # 对 labels 对齐：插入位置对应的标签置 -100
        if labels is not None:
            labels_list = []
            for b in range(labels.size(0)):
                new_len = injected["input_ids"][b].size(0)
                old = labels[b]
                pad_len = new_len - old.size(0)
                pad = old.new_full((pad_len,), -100)
                labels_list.append(torch.cat([old, pad], dim=0))
            labels = torch.stack(labels_list, dim=0)

        # 6) 槽位/答案生成
        gen_flag = mode != "sft"
        slot_out = self.slot_reasoner(
            input_ids=injected["input_ids"] if not gen_flag else None,
            attention_mask=injected["attention_mask"],
            inputs_embeds=injected["inputs_embeds"],
            labels=labels,
            generate=gen_flag,
            max_new_tokens=self.slot_reasoner.max_answer_len,
            **(generate_kwargs or {}),
        )

        return {
            "segmentation": seg_out,
            "focus_map": focus_map,
            "evidence": evidence,
            "slot": slot_out,
        }
