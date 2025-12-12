# -*- coding: utf-8 -*-
"""
Batch collate：将样本列表拼成批次。
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from echoreason.constants import SEG_TOKEN


class BatchCollator:
    """
    stage:
      - "sft"  : 监督微调（保留 question/explanation/answer）
      - "grpo" : 只需 question（用于策略生成），保留 mask 计算奖励
      - "infer": 仅推理
    tokenizer: 可选，若提供则会对 prompt/target 做编码。
    """

    def __init__(
        self,
        stage: str = "sft",
        tokenizer: Optional[Any] = None,
        max_length: int = 1024,
    ) -> None:
        self.stage = stage
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _build_texts(self, sample: Dict[str, Any]) -> Dict[str, str]:
        question = sample.get("question", "")
        explanation = sample.get("explanation", None)
        answer = sample.get("answer", None)

        # 统一在 prompt 前插入 seg token，确保分割查询能命中指定 token
        if SEG_TOKEN not in question:
            question = f"{SEG_TOKEN} {question}".strip()

        evidence = ""
        if explanation:
            evidence = (
                "<EVIDENCE>\n"
                f"[function] {explanation.function}\n"
                f"[structure] {explanation.structure}\n"
                f"[context] {explanation.context}\n"
                "</EVIDENCE>"
            )
        target_text = evidence
        if answer:
            target_text = (target_text + "\n" + answer).strip()

        return {"prompt": question, "target": target_text}

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            return {}
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [b["image"] for b in batch]
        masks = [b.get("mask", None) for b in batch]

        image_tensor = torch.stack(images, dim=0)  # [B,3,H,W]

        mask_tensor = None
        if any(m is not None for m in masks):
            max_h = max(m.shape[0] for m in masks if m is not None)
            max_w = max(m.shape[1] for m in masks if m is not None)
            padded = []
            for m in masks:
                if m is None:
                    padded.append(torch.zeros(max_h, max_w, dtype=torch.long))
                else:
                    h, w = m.shape
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded.append(
                        F.pad(
                            m,
                            (0, pad_w, 0, pad_h),
                            value=0,
                        )
                    )
            mask_tensor = torch.stack(padded, dim=0)  # [B,H,W]

        prompts = []
        targets = []
        metas: List[Dict[str, Any]] = []
        for b in batch:
            text_dict = self._build_texts(b)
            prompts.append(text_dict["prompt"])
            targets.append(text_dict["target"])
            metas.append(
                {
                    "task_type": b.get("task_type"),
                    "class_id": b.get("class_id"),
                    "class_name": b.get("class_name"),
                    "language": b.get("language"),
                    "image_id": b.get("image_id"),
                }
            )

        tokenized = self._tokenize(prompts)

        batch_dict: Dict[str, Any] = {
            "pixel_values": image_tensor,
            "masks": mask_tensor,
            "prompts": prompts,
            "targets": targets,
            "metas": metas,
        }
        batch_dict.update(tokenized)
        return batch_dict
