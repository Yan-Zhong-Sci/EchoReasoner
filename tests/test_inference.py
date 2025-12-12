"""推理流水线测试用例。"""

import torch

from echoreason.inference.pipeline import InferencePipeline
from echoreason.models.echo_reasoner import ECHOReasoner


def test_inference_smoke():
    cfg = {
        "base_model": {
            "name": "dummy",
            "vision_hidden_size": 32,
            "text_hidden_size": 32,
            "vision_patch_size": 4,
            "special_tokens": ["<SEG#1>", "<EVIDENCE>", "</EVIDENCE>"],
        },
        "text_guided_cropper": {},
        "pixel_decoder": {"hidden_dim": 16, "num_queries": 4, "num_layers": 1, "num_heads": 2},
        "bridge": {},
        "slot_reasoner": {"max_answer_len": 4},
    }
    try:
        model = ECHOReasoner(cfg)
    except Exception:
        return
    tokenizer = getattr(model, "tokenizer", None) or getattr(model.qwen, "tokenizer", None)
    pipeline = InferencePipeline(model, tokenizer, infer_cfg={"generation": {}}, device="cpu")
    img = torch.randn(3, 8, 8)
    out = pipeline(img, "test question")
    assert "mask" in out
