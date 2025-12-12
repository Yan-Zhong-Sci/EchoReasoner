"""可视化：原图叠加掩膜、槽位证据标注导出。"""

from typing import List, Optional

import numpy as np
from PIL import Image


def overlay_mask(
    image: np.ndarray,  # [H,W,3], float32 in [0,1]
    mask: np.ndarray,   # [H,W], float32 in [0,1]
    color=(255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[..., 0] = color[0] / 255.0
    colored[..., 1] = color[1] / 255.0
    colored[..., 2] = color[2] / 255.0
    mask_exp = mask[..., None]
    overlay = overlay * (1 - alpha * mask_exp) + colored * (alpha * mask_exp)
    return overlay


def save_visualization(
    image: np.ndarray,         # [H,W,3] float32 [0,1]
    mask: np.ndarray,          # [H,W] float32 [0,1]
    slot_texts: Optional[List[str]],
    out_path: str,
) -> None:
    vis = overlay_mask(image, mask)
    vis = (vis * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(vis)
    if slot_texts:
        text = "\n".join(slot_texts)
        # 简单保存文本到同目录
        txt_path = out_path + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    img.save(out_path)
