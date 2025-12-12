"""数据与图像/掩膜的读写工具。"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_mask(mask: torch.Tensor, path: str) -> None:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    img = Image.fromarray(arr.astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_image(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
