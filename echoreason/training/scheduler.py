"""学习率调度：warmup + cosine/step 等。"""

from typing import Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR


def build_scheduler(optimizer: Optimizer, cfg: Dict, total_steps: int):
    sch_cfg = cfg.get("scheduler", {})
    sch_type = sch_cfg.get("type", "cosine")
    warmup_steps = int(sch_cfg.get("warmup_steps", 0))

    def warmup_lambda(step):
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    if sch_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=0.0,
        )
    elif sch_type == "step":
        step_size = int(sch_cfg.get("step_size", max(1, total_steps // 3)))
        gamma = float(sch_cfg.get("gamma", 0.1))
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        main_scheduler = None

    if warmup_steps > 0:
        warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        return {"warmup": warmup, "main": main_scheduler}
    return {"warmup": None, "main": main_scheduler}
