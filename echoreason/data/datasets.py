# -*- coding: utf-8 -*-
"""
数据集实现，支持 JSON 描述，路径可为相对路径（配合 data_root）或绝对路径。
[
  {
    "image_id": "...",
    "image": {"path": "...", "height": 1024, "width": 1024},
    "mask": {"path": "...", "class_id": 2, "role": "main"},
    "task_type": "facility",
    "language": "en",
    "class_id": 2,
    "class_name": "Harbor/Port",
    "question": "...",
    "explanation": {"function": "...", "structure": "...", "context": "..."},
    "answer": "..."
  },
  ...
]
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .schemas import ImageInfo, MaskInfo, Sample, Explanation
from .transforms import load_image, load_mask


def _load_samples(
    annotation_path: str,
    task_filter: Optional[str] = None,
) -> List[Sample]:
    """
    读取注释：
      - 若为文件，则按单个 JSON 文件读取（列表格式）；
      - 若为目录，则读取目录下所有 .json 文件并合并。
    """
    p = Path(annotation_path)
    json_files: List[Path] = []
    if p.is_dir():
        json_files = sorted(p.glob("*.json"))
    else:
        json_files = [p]

    data: List[Dict[str, Any]] = []
    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                data.extend(content)
            else:
                raise ValueError(f"[Dataset] JSON file must contain a list: {jf}")

    samples: List[Sample] = []
    for item in data:
        if task_filter and item.get("task_type") != task_filter:
            continue

        image_dict = item.get("image", {})
        mask_dict = item.get("mask", {})
        expl_dict = item.get("explanation", None)

        image = ImageInfo(
            path=image_dict["path"],
            height=int(image_dict.get("height", 0)),
            width=int(image_dict.get("width", 0)),
        )
        masks = [
            MaskInfo(
                path=mask_dict["path"],
                class_id=int(mask_dict.get("class_id", -1)),
                role=str(mask_dict.get("role", "main")),
            )
        ]
        explanation = None
        if expl_dict:
            explanation = Explanation(
                function=expl_dict.get("function", ""),
                structure=expl_dict.get("structure", ""),
                context=expl_dict.get("context", ""),
            )

        sample = Sample(
            image_id=item.get("image_id", ""),
            image=image,
            masks=masks,
            task_type=item.get("task_type", "unknown"),
            language=item.get("language", "en"),
            question=item.get("question", ""),
            class_id=int(item.get("class_id", -1)),
            class_name=item.get("class_name", ""),
            explanation=explanation,
            answer=item.get("answer", None),
        )
        samples.append(sample)
    return samples


class BaseSegReasonDataset(Dataset):
    """
    读取 JSON -> dict:
      {
        "image": Tensor[C,H,W],
        "mask": Tensor[H,W] or None,
        "question": str,
        "explanation": Explanation or None,
        "answer": str or None,
        "meta": {image_id, task_type, class_id, class_name, language}
      }
    transform: 可选 callable(sample_dict) -> sample_dict
    支持相对路径：提供 data_root 时会与 image/mask 的 path 拼接；绝对路径将直接使用。
    """

    def __init__(
        self,
        annotation_file: str,
        task_filter: Optional[str] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.samples = _load_samples(annotation_file, task_filter=task_filter)
        if len(self.samples) == 0:
            raise ValueError(f"[Dataset] No samples found in {annotation_file} with task={task_filter}.")
        self.transform = transform
        self.data_root = Path(data_root) if data_root else None

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_str: str) -> str:
        """
        支持发布：JSON 可存相对路径/文件名，若 data_root 提供则拼接；绝对路径直接返回。
        """
        p = Path(path_str)
        if p.is_absolute() or self.data_root is None:
            return str(p)
        return str(self.data_root / p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image_path = self._resolve_path(sample.image.path)
        mask_path = self._resolve_path(sample.masks[0].path) if sample.masks else None

        image = load_image(image_path)
        mask = load_mask(mask_path) if mask_path is not None else None

        # 默认按样本的 class_id 做二值掩码：掩膜只保留目标类，其余视为背景
        if mask is not None:
            target_class = sample.class_id
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            mask = (mask.long() == int(target_class)).long()  # [H,W] 0/1

        item: Dict[str, Any] = {
            "image": image,              # Tensor[C,H,W]
            "mask": mask,                # Tensor[H,W] or None
            "question": sample.question,
            "explanation": sample.explanation,
            "answer": sample.answer,
            "task_type": sample.task_type,
            "class_id": sample.class_id,
            "class_name": sample.class_name,
            "language": sample.language,
            "image_id": sample.image_id,
        }

        if self.transform is not None:
            item = self.transform(item)
        return item


class FacilitySegReasonDataset(BaseSegReasonDataset):
    def __init__(
        self,
        annotation_file: str,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_root: Optional[str] = None,
    ) -> None:
        super().__init__(annotation_file, task_filter="facility", transform=transform, data_root=data_root)


class RegionSegReasonDataset(BaseSegReasonDataset):
    def __init__(
        self,
        annotation_file: str,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_root: Optional[str] = None,
    ) -> None:
        super().__init__(annotation_file, task_filter="region", transform=transform, data_root=data_root)


class DisasterDecisionDataset(BaseSegReasonDataset):
    def __init__(
        self,
        annotation_file: str,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_root: Optional[str] = None,
    ) -> None:
        super().__init__(annotation_file, task_filter="disaster", transform=transform, data_root=data_root)
