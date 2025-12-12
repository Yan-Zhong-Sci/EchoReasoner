"""杂项工具：计时器、配置打印、参数统计。"""

import time
from typing import Any, Dict


class Timer:
    def __init__(self):
        self.start_t = time.time()

    def reset(self):
        self.start_t = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_t


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_config(cfg: Dict[str, Any]) -> None:
    import pprint

    pprint.pprint(cfg)
