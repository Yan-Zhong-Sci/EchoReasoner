"""分割/槽位/答案的监督损失实现（含深监督）。"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims) + eps
    dice = 2.0 * intersection / union
    return 1.0 - dice.mean()


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def segmentation_loss(
    pred_logits: torch.Tensor,        # [B,K,H,W] or [B,1,H,W]
    gt_masks: torch.Tensor,           # [B,H,W] (binary 0/1)
    pred_scores: Optional[torch.Tensor] = None,  # [B,K]
    dice_weight: float = 5.0,
    focal_weight: float = 20.0,
    score_weight: float = 1.0,
    suppress_aux: bool = True,
    aux_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Query0 绑定 GT 掩膜：Dice + Focal。
    可选对其他 query 加入背景抑制。
    """
    B = gt_masks.size(0)
    Ht, Wt = gt_masks.shape[-2:]
    if pred_logits.dim() == 3:  # [B,H,W]
        pred_logits = pred_logits.unsqueeze(1)

    pred_main = pred_logits[:, 0]  # 绑定第 0 个 query
    aux_logits = pred_logits[:, 1:] if pred_logits.size(1) > 1 else None

    if pred_main.shape[-2:] != (Ht, Wt):
        pred_main = F.interpolate(
            pred_main.unsqueeze(1),
            size=(Ht, Wt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    gt_float = gt_masks.float()
    loss_dice = dice_loss_from_logits(pred_main, gt_float)
    loss_focal = sigmoid_focal_loss(pred_main, gt_float)

    losses: Dict[str, torch.Tensor] = {
        "total": dice_weight * loss_dice + focal_weight * loss_focal,
        "dice": loss_dice,
        "focal": loss_focal,
    }

    # 分类分数监督：Query0 为正类，其余为负类
    if pred_scores is not None:
        target_scores = torch.zeros_like(pred_scores)
        target_scores[:, 0] = 1.0
        loss_score = F.binary_cross_entropy_with_logits(pred_scores, target_scores)
        losses["score"] = loss_score
        losses["total"] = losses["total"] + score_weight * loss_score

    if suppress_aux and aux_logits is not None and aux_logits.numel() > 0:
        aux = aux_logits
        if aux.shape[-2:] != (Ht, Wt):
            aux = F.interpolate(aux, size=(Ht, Wt), mode="bilinear", align_corners=False)
        target_bg = torch.zeros_like(aux)
        aux_loss = sigmoid_focal_loss(aux, target_bg)
        losses["aux_bg"] = aux_weight * aux_loss
        losses["total"] = losses["total"] + losses["aux_bg"]

    return losses


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    汇总分割与文本部分的损失。文本部分直接使用 slot_reasoner 返回的 loss。
    weights: {"seg": w_seg, "slot": w_slot}
    """
    w = {"seg": 1.0, "slot": 1.0}
    if weights:
        w.update(weights)
    # 细粒度权重（若未提供则使用默认值）
    dice_w = float(w.get("dice", 5.0))
    focal_w = float(w.get("focal", 20.0))
    score_w = float(w.get("score", 1.0))

    losses: Dict[str, torch.Tensor] = {}

    seg_out = outputs.get("segmentation")
    gt_masks = batch.get("masks", None)
    seg_total = torch.tensor(0.0, device=gt_masks.device if gt_masks is not None else "cpu")

    if seg_out is not None and gt_masks is not None:
        # 主输出
        seg_losses = segmentation_loss(
            pred_logits=seg_out["logits"],
            gt_masks=gt_masks,
            pred_scores=seg_out.get("scores"),
            dice_weight=dice_w,
            focal_weight=focal_w,
            score_weight=score_w,
        )
        seg_total = seg_total + seg_losses["total"]
        losses.update({f"seg_main_{k}": v for k, v in seg_losses.items()})

        # 深层监督：遍历 aux 列表，结构应与主输出一致（包含 logits/scores）
        aux_list = seg_out.get("aux", None)
        if aux_list is not None:
            for i, aux_out in enumerate(aux_list):
                if not isinstance(aux_out, dict) or "logits" not in aux_out:
                    continue
                aux_losses = segmentation_loss(
                    pred_logits=aux_out["logits"],
                    gt_masks=gt_masks,
                    pred_scores=aux_out.get("scores"),
                    dice_weight=dice_w,
                    focal_weight=focal_w,
                    score_weight=score_w,
                )
                seg_total = seg_total + aux_losses["total"]
                losses[f"seg_aux{i}_total"] = aux_losses["total"]

    losses["seg"] = seg_total

    slot_out = outputs.get("slot", {})
    slot_loss = slot_out.get("loss", None) if isinstance(slot_out, dict) else None
    if slot_loss is None:
        slot_loss = torch.tensor(0.0, device=seg_total.device)
    losses["slot"] = slot_loss

    losses["total"] = w["seg"] * losses["seg"] + w["slot"] * losses["slot"]
    return losses
