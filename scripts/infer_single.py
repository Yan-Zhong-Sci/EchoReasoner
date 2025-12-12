"""单样本推理入口。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from echoreason.config import load_configs
from echoreason.inference.pipeline import InferencePipeline
from echoreason.inference.visualizer import save_visualization
from echoreason.models.echo_reasoner import ECHOReasoner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=".", help="项目根目录（包含 configs/）")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main():
    args = parse_args()
    cfg = load_configs(base_dir=args.config_dir)

    model = ECHOReasoner(cfg["model"])
    tokenizer = model.qwen.tokenizer
    pipeline = InferencePipeline(model, tokenizer, cfg["infer"], device=args.device)

    image = load_image(args.image)
    result = pipeline(image, args.question)

    mask = result["mask"].numpy()
    img = np.array(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    slot_texts = result.get("slot_texts", [])
    save_visualization(img, mask, slot_texts, args.output)
    print(f"Saved visualization to {args.output}")
    if slot_texts:
        print("Slot texts:")
        for t in slot_texts:
            print(t)


if __name__ == "__main__":
    main()
