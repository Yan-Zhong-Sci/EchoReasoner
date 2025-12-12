"""推理流程：单图支持 tiling、分割、证据回灌、文本生成。"""

from typing import Any, Dict, Optional

import torch

from echoreason.data.tiling import merge_tile_masks, tile_image
from echoreason.constants import SEG_TOKEN


class InferencePipeline:
    def __init__(
        self,
        model,
        tokenizer,
        infer_cfg: Dict[str, Any],
        device: str = "cuda",
        tile_size: int = 1024,
        stride: int = 768,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = infer_cfg
        self.device = device
        self.tile_size = tile_size
        self.stride = stride
        norm_cfg = infer_cfg.get("normalize", {}) if isinstance(infer_cfg, dict) else {}
        self.normalize_mean = torch.tensor(norm_cfg.get("mean", [0.5, 0.5, 0.5])).view(3, 1, 1)
        self.normalize_std = torch.tensor(norm_cfg.get("std", [0.5, 0.5, 0.5])).view(3, 1, 1)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, question: str, mask_gt: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        image: [3,H,W] tensor，未归一化情况下需要自行处理。
        """
        H, W = image.shape[-2:]
        # 推理阶段也统一插入 seg token，保持与训练一致
        if SEG_TOKEN not in question:
            question = f"{SEG_TOKEN} {question}".strip()
        tiles = tile_image(image, mask=mask_gt, tile_size=self.tile_size, stride=self.stride)
        tile_results = []
        use_tta = bool(self.cfg.get("use_tta", False))

        def _normalize(img: torch.Tensor) -> torch.Tensor:
            # 如果像素未归一化，做一次 0~1 缩放和标准化
            if img.max() > 1.5:
                img = img / 255.0
            return (img - self.normalize_mean.to(img.device)) / self.normalize_std.to(img.device)

        def _forward_once(img: torch.Tensor):
            """单次前向，返回掩膜概率和槽位输出。"""
            img = _normalize(img)
            input_ids = self.tokenizer(
                [question], padding=True, truncation=True, max_length=1024, return_tensors="pt"
            )
            outputs = self.model(
                {
                    "pixel_values": img.unsqueeze(0),
                    "input_ids": input_ids.input_ids.to(self.device),
                    "attention_mask": input_ids.attention_mask.to(self.device),
                },
                mode="infer",
                generate_kwargs={
                    "temperature": self.cfg.get("generation", {}).get("temperature", 0.7),
                    "top_p": self.cfg.get("generation", {}).get("top_p", 0.9),
                    "do_sample": False,
                    "max_new_tokens": self.cfg.get("generation", {}).get("max_new_tokens", 160),
                },
            )
            seg_out = outputs["segmentation"]
            pred_masks = seg_out["masks"]  # [B,K,Ht,Wt], 已经是概率
            if pred_masks.size(1) > 1:
                idx = seg_out["scores"].argmax(dim=1)
                pred_masks = torch.stack(
                    [pred_masks[b, idx[b]] for b in range(pred_masks.size(0))], dim=0
                )
            else:
                pred_masks = pred_masks[:, 0]
            return pred_masks, outputs["slot"]

        for t in tiles:
            img = t["image"].to(self.device)
            pred_masks, slot_out = _forward_once(img)

            if use_tta:
                img_flip = torch.flip(img, dims=[-1])
                pred_flip, _ = _forward_once(img_flip)
                pred_flip = torch.flip(pred_flip, dims=[-1])
                pred_masks = (pred_masks + pred_flip) * 0.5

            tile_results.append(
                {
                    "mask": pred_masks[0].detach().cpu(),
                    "offset": t["offset"],
                    "slot": slot_out,
                }
            )

        merged_mask = merge_tile_masks(tile_results, image_size=(H, W), reduce="max")
        slot_texts = tile_results[0]["slot"].get("texts", []) if tile_results else []

        # 阈值与最小面积
        mask_prob_thresh = self.cfg.get("mask_prob_thresh", 0.5)
        min_area = int(self.cfg.get("min_area", 64))
        bin_mask = (merged_mask >= mask_prob_thresh).float()
        if bin_mask.sum() < min_area:
            bin_mask.zero_()

        return {
            "mask": bin_mask,
            "mask_prob": merged_mask,
            "slot_texts": slot_texts,
        }
