"""优化器构建：按 LoRA / 基座 / 下游拆分学习率与权重衰减。"""

from typing import Dict, Iterable, List, Tuple

import torch


def _param_group(params: Iterable, lr: float, weight_decay: float) -> Dict:
    return {"params": list(params), "lr": lr, "weight_decay": weight_decay}


def build_optimizer(model, cfg: Dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 3e-5))
    wd = float(opt_cfg.get("weight_decay", 0.01))

    lr_base = float(opt_cfg.get("lr_base", base_lr))
    lr_pixel = float(opt_cfg.get("lr_pixel", base_lr))
    lr_bridge = float(opt_cfg.get("lr_bridge", base_lr))
    lr_slot = float(opt_cfg.get("lr_slot", base_lr))
    lr_lora = float(opt_cfg.get("lora_lr", base_lr))
    wd_lora = float(opt_cfg.get("lora_weight_decay", 0.0))

    groups: List[Dict] = []
    used = set()

    def add(params, lr, decay):
        params = [p for p in params if p.requires_grad and id(p) not in used]
        for p in params:
            used.add(id(p))
        if params:
            groups.append(_param_group(params, lr, decay))

    # 先将 LoRA 参数拆组，赋予独立学习率 / 衰减
    qwen_named: List[Tuple[str, torch.nn.Parameter]] = list(model.qwen.model.named_parameters())
    lora_params = [p for n, p in qwen_named if "lora_" in n and p.requires_grad]
    add(lora_params, lr_lora, wd_lora)

    # Qwen 基座（非 LoRA），可能通过 tuning_modules 解冻
    non_lora_qwen = [p for n, p in qwen_named if "lora_" not in n]
    add(non_lora_qwen, lr_base, wd)

    # 下游模块
    add(model.segmentation.parameters(), lr_pixel, wd)
    add(model.bridge.parameters(), lr_bridge, wd)
    add(model.slot_reasoner.parameters(), lr_slot, wd)

    optimizer = torch.optim.AdamW(groups, betas=tuple(opt_cfg.get("betas", (0.9, 0.98))))
    return optimizer
