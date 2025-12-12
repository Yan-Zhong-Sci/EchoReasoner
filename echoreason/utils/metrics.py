"""分割、槽位与文本指标。"""

import torch
import torch.nn.functional as F


class ConfusionMatrix:
    def __init__(self, num_classes: int = 2) -> None:
        self.mat = None
        self.n = num_classes

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """
        pred, gt: flat tensors of same length
        """
        if self.mat is None:
            self.mat = torch.zeros((self.n, self.n), device=pred.device)
        k = (gt >= 0) & (gt < self.n)
        inds = self.n * gt[k] + pred[k]
        self.mat += torch.bincount(inds, minlength=self.n**2).reshape(self.n, self.n)

    def compute_miou(self) -> float:
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-6)
        return iu.mean().item()


def seg_iou(pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(1)
    pred = torch.sigmoid(pred_logits[:, 0])
    pred = F.interpolate(pred.unsqueeze(1), size=gt_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pred_bin = (pred >= 0.5).long()
    cm = ConfusionMatrix(num_classes=2)
    for p, g in zip(pred_bin, gt_mask.long()):
        cm.update(p.flatten(), g.flatten())
    return torch.tensor(cm.compute_miou())


def seg_dice(pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(1)
    pred = torch.sigmoid(pred_logits[:, 0])
    pred = F.interpolate(pred.unsqueeze(1), size=gt_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    inter = (pred * gt_mask).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt_mask.sum(dim=(-1, -2))
    dice = 2 * inter / (union + 1e-6)
    return dice.mean()


def text_exact_match(pred_texts, target_texts):
    scores = []
    for p, t in zip(pred_texts, target_texts):
        scores.append(1.0 if p.strip() == (t or "").strip() and p.strip() != "" else 0.0)
    return torch.tensor(scores, dtype=torch.float32)
