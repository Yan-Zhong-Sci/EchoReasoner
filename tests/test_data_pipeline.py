"""数据读取/tiling/collate 测试用例。"""

import torch

from echoreason.data.collate import BatchCollator


def test_collate_shapes():
    collate = BatchCollator(stage="sft", tokenizer=None, max_length=16)
    sample = {
        "image": torch.randn(3, 4, 4),
        "mask": torch.zeros(4, 4, dtype=torch.long),
        "question": "where is target?",
        "explanation": None,
        "answer": "none",
        "task_type": "facility",
        "class_id": 1,
        "class_name": "test",
        "language": "en",
        "image_id": "id",
    }
    batch = collate([sample, sample])
    assert batch["pixel_values"].shape[0] == 2
