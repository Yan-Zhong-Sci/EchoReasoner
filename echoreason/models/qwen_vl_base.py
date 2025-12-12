# -*- coding: utf-8 -*-
"""
Qwen-VL base wrapper.

职责：
  - 加载 Qwen2.5-VL（3B/7B 等）的 tokenizer/processor/model；
  - 增补特殊 tokens（<SEG#1>/<EVIDENCE>/</EVIDENCE>），同步到模型嵌入；
  - 提供文本/视觉隐状态及多层视觉特征；
  - 生成多尺度金字塔供像素解码器使用；
说明：
  - 不改动基座内部结构，只做封装；
  - forward 需输入已对齐的 pixel_values（可用 AutoProcessor 预处理）。
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


class QwenVLBase(nn.Module):
    def __init__(
        self,
        model_name: str,
        vision_hidden_size: int,
        text_hidden_size: int,
        vision_patch_size: int = 14,
        special_tokens: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        trust_remote_code: bool = True,
        load_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # 兼容旧字段
        lora_target_modules: Optional[List[str]] = None,
        lora_layer_type: str = "qkvo",
        lora_modules_scope: str = "all",
        # 新字段：文本/视觉各自独立的 LoRA 策略
        lora_layer_type_llm: str = "qkvo",       # "qkvo" | "linear"
        lora_layer_type_visual: str = "qkvo",    # "qkvo" | "linear"
        lora_r_llm: Optional[int] = None,
        lora_r_visual: Optional[int] = None,
        lora_alpha_llm: Optional[int] = None,
        lora_alpha_visual: Optional[int] = None,
        lora_dropout_llm: Optional[float] = None,
        lora_dropout_visual: Optional[float] = None,
        lora_target_modules_llm: Optional[List[str]] = None,
        lora_target_modules_visual: Optional[List[str]] = None,
        lora_include_llm: bool = True,
        lora_include_visual: bool = True,
        vision_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        self.vision_hidden_size = int(vision_hidden_size)
        self.text_hidden_size = int(text_hidden_size)
        self.vision_patch_size = int(vision_patch_size)
        self.special_token_ids: Dict[str, int] = {}
        self.vision_layers = vision_layers if vision_layers is not None else [-4, -3, -2, -1]

        # LoRA 配置
        self.lora_layer_type_llm = str(lora_layer_type_llm or lora_layer_type)
        self.lora_layer_type_visual = str(lora_layer_type_visual or lora_layer_type)
        self.lora_target_modules_llm = lora_target_modules_llm
        self.lora_target_modules_visual = lora_target_modules_visual
        self.lora_include_llm = bool(lora_include_llm)
        self.lora_include_visual = bool(lora_include_visual)
        # 独立超参（为空则回退到全局 lora_r/alpha/dropout）
        self.lora_r_llm = lora_r_llm
        self.lora_r_visual = lora_r_visual
        self.lora_alpha_llm = lora_alpha_llm
        self.lora_alpha_visual = lora_alpha_visual
        self.lora_dropout_llm = lora_dropout_llm
        self.lora_dropout_visual = lora_dropout_visual
        # 兼容旧配置
        self._legacy_lora_target_modules = lora_target_modules
        self._legacy_lora_scope = lora_modules_scope

        if special_tokens:
            self._add_special_tokens(special_tokens)

        if load_lora:
            self._inject_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def _add_special_tokens(self, tokens: List[str]) -> None:
        to_add = [t for t in tokens if t not in self.tokenizer.get_vocab()]
        if to_add:
            added = self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            if added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
        for t in tokens:
            self.special_token_ids[t] = self.tokenizer.convert_tokens_to_ids(t)

    def _discover_lora_targets(self, layer_type: str, modules_scope: str) -> List[str]:
        """
        动态发现 LoRA 目标模块：
          - layer_type: "qkvo" 关注注意力/FFN 投影；"linear" 则所有 Linear
          - modules_scope: "llm" | "visual" | "all"
        """
        target_keywords = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "attn.qkv",
            "attn.proj",
            "gate",
            "up_proj",
            "down_proj",
            "fc",
            "mlp",
        ]
        targets: List[str] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            is_visual = "visual" in name
            if modules_scope == "llm" and is_visual:
                continue
            if modules_scope == "visual" and not is_visual:
                continue
            if layer_type == "qkvo":
                if not any(name.endswith(k) or k in name for k in target_keywords):
                    continue
            targets.append(name)
        uniq: List[str] = []
        seen = set()
        for t in targets:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq

    def _inject_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Optional[List[str]],
    ) -> None:
        targets_llm: List[str] = []
        targets_vis: List[str] = []

        # 1) 兼容旧字段：显式 lora_target_modules（作用于两侧）
        legacy_targets: List[str] = target_modules or []
        # 兼容旧 scope（无新字段时回退）
        legacy_scope = self._legacy_lora_scope

        # 2) 新字段：文本侧
        if self.lora_include_llm:
            if self.lora_target_modules_llm:
                targets_llm.extend(self.lora_target_modules_llm)
            else:
                targets_llm.extend(
                    self._discover_lora_targets(
                        layer_type=self.lora_layer_type_llm,
                        modules_scope="llm",
                    )
                )

        # 3) 新字段：视觉侧
        if self.lora_include_visual:
            if self.lora_target_modules_visual:
                targets_vis.extend(self.lora_target_modules_visual)
            else:
                targets_vis.extend(
                    self._discover_lora_targets(
                        layer_type=self.lora_layer_type_visual,
                        modules_scope="visual",
                    )
                )

        # 4) 兼容旧配置：若新字段都没抓到，则按 legacy scope 发现
        if not targets_llm and not targets_vis and legacy_scope:
            if legacy_scope in ("llm", "all"):
                targets_llm.extend(
                    self._discover_lora_targets(
                        layer_type=self.lora_layer_type_llm,
                        modules_scope="llm",
                    )
                )
            if legacy_scope in ("visual", "all"):
                targets_vis.extend(
                    self._discover_lora_targets(
                        layer_type=self.lora_layer_type_visual,
                        modules_scope="visual",
                    )
                )

        # 若仍无目标，尝试 legacy 显式列表
        if not targets_llm and not targets_vis and legacy_targets:
            targets_llm.extend(legacy_targets)

        if not targets_llm and not targets_vis:
            raise ValueError("No LoRA target modules discovered. Check LoRA configuration.")

        # 去重保持顺序
        def _dedup(seq: List[str]) -> List[str]:
            uniq: List[str] = []
            seen = set()
            for t in seq:
                if t in seen:
                    continue
                seen.add(t)
                uniq.append(t)
            return uniq

        targets_llm = _dedup(targets_llm)
        targets_vis = _dedup(targets_vis)

        # 每侧独立超参（为空则用全局）
        r_llm = int(self.lora_r_llm) if self.lora_r_llm is not None else int(r)
        r_vis = int(self.lora_r_visual) if self.lora_r_visual is not None else int(r)
        alpha_llm = int(self.lora_alpha_llm) if self.lora_alpha_llm is not None else int(alpha)
        alpha_vis = int(self.lora_alpha_visual) if self.lora_alpha_visual is not None else int(alpha)
        drop_llm = float(self.lora_dropout_llm) if self.lora_dropout_llm is not None else float(dropout)
        drop_vis = float(self.lora_dropout_visual) if self.lora_dropout_visual is not None else float(dropout)

        applied = False
        adapter_names: List[str] = []

        if targets_llm:
            cfg_llm = LoraConfig(
                r=r_llm,
                lora_alpha=alpha_llm,
                lora_dropout=drop_llm,
                target_modules=targets_llm,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, cfg_llm, adapter_name="llm")
            applied = True
            adapter_names.append("llm")

        if targets_vis:
            cfg_vis = LoraConfig(
                r=r_vis,
                lora_alpha=alpha_vis,
                lora_dropout=drop_vis,
                target_modules=targets_vis,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if applied:
                # 已是 PeftModel，添加新适配器
                self.model.add_adapter("visual", cfg_vis)
            else:
                self.model = get_peft_model(self.model, cfg_vis, adapter_name="visual")
                applied = True
            adapter_names.append("visual")

        if applied and len(adapter_names) > 1:
            # 同时启用两侧 LoRA
            self.model.set_adapter(adapter_names)

    def get_special_token_id(self, token: str) -> Optional[int]:
        return self.special_token_ids.get(token, None)

    def _reshape_vision_hidden(
        self,
        vision_hidden: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        B, N, Cv = vision_hidden.shape
        if Cv != self.vision_hidden_size:
            raise ValueError(f"vision_hidden_size {Cv} != expected {self.vision_hidden_size}")

        if image_size is not None:
            H_img, W_img = image_size
            H = max(H_img // self.vision_patch_size, 1)
            W = max(W_img // self.vision_patch_size, 1)
        else:
            root = int(N ** 0.5)
            if root * root != N:
                raise ValueError(f"Cannot reshape vision tokens length {N} to square grid.")
            H = W = root

        if H * W != N:
            raise ValueError(f"vision_hidden length {N} incompatible with H*W={H*W}.")

        return vision_hidden.transpose(1, 2).contiguous().view(B, Cv, H, W)

    @staticmethod
    def build_pyramid(feature: torch.Tensor, extra_feats: Optional[List[torch.Tensor]] = None) -> Dict[int, torch.Tensor]:
        """
        构建多尺度特征：
          - 若提供 extra_feats（多层特征），使用浅层提升分辨率；
          - 否则回退到单层池化。
        """
        if extra_feats and len(extra_feats) > 0:
            shallow = extra_feats[0]
            # 尝试得到更高分辨率
            s4 = F.interpolate(shallow, scale_factor=2, mode="bilinear", align_corners=False)
            s8 = shallow
            s16 = F.avg_pool2d(shallow, kernel_size=2, stride=2)
            s32 = F.avg_pool2d(shallow, kernel_size=4, stride=4)
            return {4: s4, 8: s8, 16: s16, 32: s32}
        # 退化为单层池化
        s8 = feature
        s16 = F.avg_pool2d(s8, kernel_size=2, stride=2)
        s32 = F.avg_pool2d(s8, kernel_size=4, stride=4)
        return {8: s8, 16: s16, 32: s32}

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("Need output_hidden_states=True to get text hidden states.")
        text_hidden_all = outputs.hidden_states
        text_hidden = text_hidden_all[-1]
        if text_hidden.size(-1) != self.text_hidden_size:
            raise ValueError(f"text_hidden_size {text_hidden.size(-1)} != expected {self.text_hidden_size}")

        vision_hidden_states = None
        if hasattr(outputs, "vision_outputs") and outputs.vision_outputs is not None:
            vision_out = outputs.vision_outputs
            if hasattr(vision_out, "hidden_states") and vision_out.hidden_states:
                vision_hidden_states = vision_out.hidden_states
            elif hasattr(vision_out, "last_hidden_state"):
                vision_hidden_states = [vision_out.last_hidden_state]
        elif hasattr(outputs, "vision_hidden_states") and outputs.vision_hidden_states:
            vision_hidden_states = outputs.vision_hidden_states
        if vision_hidden_states is None:
            raise RuntimeError("No vision_hidden_states found.")

        # 选取多层视觉特征并还原为空间特征
        vision_maps: List[torch.Tensor] = []
        for idx in self.vision_layers:
            idx_real = idx if idx >= 0 else len(vision_hidden_states) + idx
            if idx_real < 0 or idx_real >= len(vision_hidden_states):
                continue
            vm = self._reshape_vision_hidden(vision_hidden_states[idx_real], image_size=pixel_values.shape[-2:])
            vision_maps.append(vm)
        # 使用最后一层作为默认 vision_map
        vision_map = vision_maps[-1] if vision_maps else self._reshape_vision_hidden(
            vision_hidden_states[-1], image_size=pixel_values.shape[-2:]
        )

        return {
            "logits": outputs.logits,
            "text_hidden": text_hidden,
            "text_hidden_all": text_hidden_all[-3:] if text_hidden_all is not None else None,
            "vision_hidden": vision_hidden_states[-1],
            "vision_map": vision_map,
            "vision_maps": vision_maps,
            "outputs": outputs,
        }

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding_layer = self.model.get_input_embeddings()
        return embedding_layer(input_ids)
