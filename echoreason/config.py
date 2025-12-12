"""统一加载/合并 configs 下 yaml 的配置入口。"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(
    base_dir: Optional[os.PathLike] = None,
    model_cfg: str = "configs/model.yaml",
    data_cfg: str = "configs/data.yaml",
    train_sft_cfg: str = "configs/train_sft.yaml",
    train_grpo_cfg: str = "configs/train_grpo.yaml",
    infer_cfg: str = "configs/infer.yaml",
) -> Dict[str, Any]:
    """
    加载并返回一个 dict，其中包含 model/data/train_sft/train_grpo/infer 五部分。
    base_dir 默认为项目根目录（含 configs/）。
    """
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    cfg = {
        "model": _load_yaml(root / model_cfg),
        "data": _load_yaml(root / data_cfg),
        "train_sft": _load_yaml(root / train_sft_cfg),
        "train_grpo": _load_yaml(root / train_grpo_cfg),
        "infer": _load_yaml(root / infer_cfg),
    }
    return cfg
