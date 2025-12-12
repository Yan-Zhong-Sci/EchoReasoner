"""SFT 训练入口脚本。"""

import argparse
import os

import torch

from echoreason.config import load_configs
from echoreason.data.datasets import FacilitySegReasonDataset
from echoreason.data.transforms import SegmentationTransform
from echoreason.models.echo_reasoner import ECHOReasoner
from echoreason.training.sft_trainer import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=".", help="项目根目录（包含 configs/）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(base_dir=args.config_dir)

    # 构建数据集（示例使用 facility_seg_reason）
    data_cfg = cfg["data"]
    tfm = SegmentationTransform()
    train_ds = None
    val_ds = None
    fac_cfg = data_cfg["datasets"]["facility_seg_reason"]
    if fac_cfg.get("enable", False):
        # 支持直接传目录：目录下默认 labels/ann.json，data_root 指向该目录
        train_dir = fac_cfg.get("train_dir", "")
        val_dir = fac_cfg.get("val_dir", "")
        default_root = data_cfg.get("data_root", None)

        if train_dir:
            train_path = os.path.join(train_dir, "labels", "ann.json")
            train_root = train_dir
        else:
            train_path = fac_cfg.get("train_ann_file") or fac_cfg.get("ann_file")
            train_root = default_root

        if train_path:
            train_ds = FacilitySegReasonDataset(annotation_file=train_path, transform=tfm, data_root=train_root)

        if val_dir:
            val_path = os.path.join(val_dir, "labels", "ann.json")
            val_root = val_dir
        else:
            val_path = fac_cfg.get("val_ann_file", "")
            val_root = default_root
        if val_path:
            val_ds = FacilitySegReasonDataset(annotation_file=val_path, transform=tfm, data_root=val_root)

    if train_ds is None:
        raise ValueError("No dataset enabled. Please set datasets.*.enable = true with valid paths.")

    # 模型与 tokenizer
    model = ECHOReasoner(cfg["model"])
    tokenizer = model.qwen.tokenizer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        cfg=cfg["train_sft"],
        device=args.device,
    )
    resume_path = cfg["train_sft"].get("resume_from", "")
    if resume_path:
        trainer.load_checkpoint(resume_path, map_location=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
