# -*- coding: utf-8 -*-
"""
大图切分与掩膜合并工具。
"""
from typing import Dict, List, Tuple

import torch


def tile_image(
    image: torch.Tensor,           # [3,H,W]
    mask: torch.Tensor = None,     # [H,W] or None
    tile_size: int = 1024,
    stride: int = 768,
) -> List[Dict[str, torch.Tensor]]:
    """
    将大图切成重叠 tile。
    返回列表，每个元素包含 {"image": tile_img, "mask": tile_mask, "offset": (y,x)}。
    """
    _, H, W = image.shape
    tiles: List[Dict[str, torch.Tensor]] = []

    ys = list(range(0, max(H - tile_size, 0) + 1, stride))
    xs = list(range(0, max(W - tile_size, 0) + 1, stride))
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    for y in ys:
        for x in xs:
            tile_img = image[:, y:y + tile_size, x:x + tile_size]
            tile_mask = None
            if mask is not None:
                tile_mask = mask[y:y + tile_size, x:x + tile_size]
            tiles.append({"image": tile_img, "mask": tile_mask, "offset": (y, x)})
    return tiles


def merge_tile_masks(
    masks_per_tile: List[Dict[str, torch.Tensor]],
    image_size: Tuple[int, int],
    reduce: str = "max",
) -> torch.Tensor:
    """
    将每个 tile 的掩膜拼回原图尺寸。
    masks_per_tile: 列表元素包含 {"mask": [K,Ht,Wt] or [Ht,Wt], "offset": (y,x)}
    reduce: "max" 使用区域最大值融合；"mean" 按重叠次数取平均。
    """
    H, W = image_size
    device = masks_per_tile[0]["mask"].device
    dtype = masks_per_tile[0]["mask"].dtype

    merged = torch.zeros((H, W), device=device, dtype=dtype)
    counter = torch.zeros((H, W), device=device, dtype=torch.float32)

    for item in masks_per_tile:
        mask = item["mask"]  # [K,Ht,Wt] 或 [Ht,Wt]
        offset_y, offset_x = item["offset"]
        if mask.dim() == 3:
            # 若是多掩膜，先取 max
            mask = mask.max(dim=0).values
        h, w = mask.shape
        if reduce == "max":
            merged_slice = merged[offset_y:offset_y + h, offset_x:offset_x + w]
            merged[offset_y:offset_y + h, offset_x:offset_x + w] = torch.maximum(merged_slice, mask)
        else:
            merged[offset_y:offset_y + h, offset_x:offset_x + w] += mask
            counter[offset_y:offset_y + h, offset_x:offset_x + w] += 1.0

    if reduce == "mean":
        merged = merged / counter.clamp_min(1.0)
    return merged
