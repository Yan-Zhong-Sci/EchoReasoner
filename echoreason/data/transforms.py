# -*- coding: utf-8 -*-
"""
图像/掩膜读取与基础增广，加入大尺度抖动（LSJ）：随机缩放再裁剪/填充到固定尺寸。
"""
import random
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # [H,W,3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    return tensor


def load_mask(path: str) -> torch.Tensor:
    mask = Image.open(path)
    arr = np.array(mask, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr)  # [H,W]


class SegmentationTransform:
    """
    大尺度抖动 + 随机裁剪/填充 + 翻转 + 归一化。
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        random_flip: bool = True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        scale_min: float = 0.5,
        scale_max: float = 2.0,
    ) -> None:
        self.image_size = image_size
        self.random_flip = random_flip
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def _random_scale(self, image: torch.Tensor, mask: torch.Tensor):
        scale = random.uniform(self.scale_min, self.scale_max)
        new_h = max(int(image.shape[1] * scale), 1)
        new_w = max(int(image.shape[2] * scale), 1)
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = F.interpolate(mask.unsqueeze(0).float(), size=(new_h, new_w), mode="nearest")[0, 0].long()
        return image, mask

    def _random_crop_pad(self, image: torch.Tensor, mask: torch.Tensor):
        target_h, target_w = self.image_size
        curr_h, curr_w = image.shape[-2:]

        # pad if needed
        pad_h = max(0, target_h - curr_h)
        pad_w = max(0, target_w - curr_w)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)
        # crop if larger
        curr_h, curr_w = image.shape[-2:]
        if curr_h > target_h or curr_w > target_w:
            top = random.randint(0, curr_h - target_h)
            left = random.randint(0, curr_w - target_w)
            image = image[:, top:top + target_h, left:left + target_w]
            if mask is not None:
                mask = mask[top:top + target_h, left:left + target_w]
        return image, mask

    def _flip(self, image: torch.Tensor, mask: torch.Tensor):
        if self.random_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转
            if mask is not None:
                mask = torch.flip(mask, dims=[1])
        return image, mask

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.normalize_mean) / self.normalize_std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]
        mask = sample.get("mask", None)

        image, mask = self._random_scale(image, mask)
        image, mask = self._random_crop_pad(image, mask)
        image, mask = self._flip(image, mask)
        image = self._normalize(image)

        sample["image"] = image
        sample["mask"] = mask
        return sample
