"""纯分割评估入口。"""

import argparse

import torch

from echoreason.config import load_configs
from echoreason.data.datasets import FacilitySegReasonDataset
from echoreason.data.transforms import SegmentationTransform
from echoreason.models.echo_reasoner import ECHOReasoner
from echoreason.utils.metrics import ConfusionMatrix
from echoreason.reward import pixel_reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=".", help="项目根目录（包含 configs/）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_configs(base_dir=args.config_dir)
    data_cfg = cfg["data"]
    tfm = SegmentationTransform()
    fac_cfg = data_cfg["datasets"]["facility_seg_reason"]
    if not fac_cfg.get("enable", False):
        raise ValueError("Enable a dataset in configs/data.yaml for evaluation.")
    ds = FacilitySegReasonDataset(annotation_file=fac_cfg["ann_file"], transform=tfm)

    model = ECHOReasoner(cfg["model"]).to(args.device)
    tokenizer = model.qwen.tokenizer

    cm = ConfusionMatrix(num_classes=2)
    total_reward = 0.0
    count = 0

    for sample in ds:
        img = sample["image"].to(args.device)
        mask_gt = (sample["mask"] > 0).long().to(args.device) if sample.get("mask") is not None else None
        if mask_gt is None:
            continue

        enc = tokenizer([sample["question"]], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        out = model(
            {
                "pixel_values": img.unsqueeze(0),
                "input_ids": enc.input_ids.to(args.device),
                "attention_mask": enc.attention_mask.to(args.device),
            },
            mode="infer",
        )
        seg_out = out["segmentation"]
        pred_logits = seg_out["logits"]  # [B,K,H,W]
        pred_scores = seg_out.get("scores", None)

        # top-1 预测用于评估
        if pred_logits.size(1) > 1:
            idx = seg_out["scores"].argmax(dim=1)
            best_logits = pred_logits[torch.arange(pred_logits.size(0)), idx]
        else:
            best_logits = pred_logits[:, 0]

        pred_prob = torch.sigmoid(best_logits)
        pred_bin = (pred_prob > 0.5).long()

        # 更新混淆矩阵
        cm.update(pred_bin.flatten(), mask_gt.flatten())

        # 兼容原有奖励统计（可选）
        r = pixel_reward(pred_logits, mask_gt.unsqueeze(0), pred_scores=pred_scores)
        total_reward += r.mean().item()
        count += 1

    miou = cm.compute_miou()
    print(f"Global mIoU: {miou:.4f}")
    if count > 0:
        print(f"Pixel reward mean: {total_reward / count:.4f}")


if __name__ == "__main__":
    main()
