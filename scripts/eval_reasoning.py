"""分割+三槽位决策综合评估入口。"""

import argparse
import os

import torch

from echoreason.config import load_configs
from echoreason.data.datasets import FacilitySegReasonDataset
from echoreason.data.transforms import SegmentationTransform
from echoreason.models.echo_reasoner import ECHOReasoner
from echoreason.reward import compose_reward, pixel_reward, text_reward


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
    data_root = data_cfg.get("data_root", None)
    fac_cfg = data_cfg["datasets"]["facility_seg_reason"]
    if not fac_cfg.get("enable", False):
        raise ValueError("Enable a dataset in configs/data.yaml for evaluation.")
    # 支持传目录：目录下默认 labels/ann.json，并将 data_root 指向该目录
    test_dir = fac_cfg.get("test_dir", "")
    if test_dir:
        test_path = os.path.join(test_dir, "labels", "ann.json")
        data_root = test_dir
    else:
        test_path = fac_cfg.get("test_ann_file") or fac_cfg.get("val_ann_file") or fac_cfg.get("ann_file")
    if not test_path:
        raise ValueError("Please provide test_ann_file (or val_ann_file) in configs/data.yaml for evaluation.")
    ds = FacilitySegReasonDataset(annotation_file=test_path, transform=tfm, data_root=data_root)

    model = ECHOReasoner(cfg["model"]).to(args.device)
    tokenizer = model.qwen.tokenizer

    pixel_scores = []
    text_scores = []
    for sample in ds:
        img = sample["image"].to(args.device)
        mask_gt = (sample["mask"] > 0).long().to(args.device) if sample.get("mask") is not None else None
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
        pred_logits = seg_out["logits"]
        pred_scores = seg_out.get("scores", None)
        if mask_gt is not None:
            pr = pixel_reward(pred_logits, mask_gt.unsqueeze(0), pred_scores=pred_scores)
            pixel_scores.append(pr.mean().item())

        slot_texts = out["slot"].get("texts", [""])
        target_text = sample.get("answer", "") or ""
        tr = text_reward(
            slot_texts,
            [target_text],
            tokenizer=tokenizer,
            embedding_layer=model.qwen.model.get_input_embeddings(),
        )
        text_scores.append(tr.mean().item())

    pixel_mean = sum(pixel_scores) / max(len(pixel_scores), 1)
    text_mean = sum(text_scores) / max(len(text_scores), 1)
    total = compose_reward(
        torch.tensor([pixel_mean]), torch.tensor([text_mean]), cfg["train_grpo"].get("reward_weight", {})
    ).mean().item()
    print(f"Pixel mean: {pixel_mean:.4f}, Text mean: {text_mean:.4f}, Composed: {total:.4f}")


if __name__ == "__main__":
    main()
