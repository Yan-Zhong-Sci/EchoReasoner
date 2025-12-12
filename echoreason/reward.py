"""GRPO 奖励定义与组合：格式 + 槽位语义 + 关键词。"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import KEYWORDS_STRUCTURE, KEYWORDS_CONTEXT, KEYWORDS_FUNCTION


def _binarize(mask: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (mask >= thresh).float()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_bin = _binarize(pred)
    target_bin = _binarize(target)
    inter = (pred_bin * target_bin).sum(dim=(-2, -1))
    union = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1)) - inter + eps
    return inter / union


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_bin = _binarize(pred)
    target_bin = _binarize(target)
    inter = (pred_bin * target_bin).sum(dim=(-2, -1))
    denom = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1)) + eps
    return 2.0 * inter / denom


def _extract_slots(text: str) -> Tuple[str, str, str]:
    """提取 [function]/[structure]/[context] 段，缺失则为空。"""
    m = re.search(r"\[function\](.*?)\[structure\](.*?)\[context\](.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return "", "", ""
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def _token_overlap(a: str, b: str) -> float:
    """简单词重叠 F1，兜底避免完全跑偏。"""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    prec = inter / len(ta)
    rec = inter / len(tb)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _keyword_hit_rate(text: str, keywords: List[str]) -> float:
    if not text or not keywords:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in keywords if k.lower() in t)
    return min(hits / len(keywords), 1.0)


def _semantic_cosine(
    pred: str,
    tgt: str,
    tokenizer: Optional[Any],
    embedding_layer: Optional[torch.nn.Module],
    max_length: int = 128,
) -> float:
    """使用模型自带 embedding 求平均向量，计算余弦相似度。"""
    if tokenizer is None or embedding_layer is None or not pred or not tgt:
        return 0.0
    with torch.no_grad():
        enc_p = tokenizer(
            pred,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc_t = tokenizer(
            tgt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = embedding_layer.weight.device
        ids_p = enc_p.input_ids.to(device)
        ids_t = enc_t.input_ids.to(device)
        if ids_p.numel() == 0 or ids_t.numel() == 0:
            return 0.0
        emb_p = embedding_layer(ids_p).mean(dim=1)  # [1,D]
        emb_t = embedding_layer(ids_t).mean(dim=1)  # [1,D]
        denom = (emb_p.norm(dim=1) * emb_t.norm(dim=1)).clamp_min(1e-6)
        cos = (emb_p * emb_t).sum(dim=1) / denom
        return float(cos.item())


def pixel_reward(
    pred_logits: torch.Tensor,    # [B,K,H,W]
    gt_mask: torch.Tensor,        # [B,H,W]
    pred_scores: torch.Tensor = None,  # [B,K]
    thresh: float = 0.5,
) -> torch.Tensor:
    """
    取最高分 query 的掩膜，计算 IoU + Dice，返回两者平均作为 pixel 奖励。
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(1)
    if pred_scores is not None and pred_logits.size(1) > 1:
        idx = pred_scores.argmax(dim=1)
        preds = torch.stack([pred_logits[b, idx[b]] for b in range(pred_logits.size(0))], dim=0)
    else:
        preds = pred_logits[:, 0]
    probs = torch.sigmoid(preds)
    probs = F.interpolate(probs.unsqueeze(1), size=gt_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    iou = iou_score(probs, gt_mask.float(), eps=1e-6)
    dice = dice_score(probs, gt_mask.float(), eps=1e-6)
    return (iou + dice) * 0.5


def text_reward(
    pred_texts: List[str],
    target_texts: List[str],
    tokenizer: Optional[Any] = None,
    embedding_layer: Optional[torch.nn.Module] = None,
    max_length: int = 128,
) -> torch.Tensor:
    """
    文本奖励：格式 + 槽位语义（余弦相似度）+ 槽位关键词 + 词重叠。
    """
    scores = []
    for pred, tgt in zip(pred_texts, target_texts):
        # 完全匹配直接满分
        if pred.strip() == (tgt or "").strip() and pred.strip():
            scores.append(1.0)
            continue

        has_evidence = ("<EVIDENCE>" in pred) and ("</EVIDENCE>" in pred)
        slots_pattern_ok = bool(
            re.search(r"\[function\].*?\[structure\].*?\[context\]", pred, re.IGNORECASE | re.DOTALL)
        )

        pf, ps, pc = _extract_slots(pred)
        tf, ts, tc = _extract_slots(tgt or "")

        slot_scores = []
        slot_defs = [
            (pf, tf, KEYWORDS_FUNCTION),
            (ps, ts, KEYWORDS_STRUCTURE),
            (pc, tc, KEYWORDS_CONTEXT),
        ]
        for p_slot, t_slot, kw in slot_defs:
            cos = _semantic_cosine(p_slot, t_slot, tokenizer, embedding_layer, max_length=max_length)
            kw_hit = _keyword_hit_rate(p_slot, kw)
            overlap = _token_overlap(p_slot, t_slot)
            slot_scores.append(0.6 * cos + 0.2 * kw_hit + 0.2 * overlap)

        slot_mean = sum(slot_scores) / len(slot_scores)

        fmt_bonus = 0.0
        if has_evidence:
            fmt_bonus += 0.1
        if slots_pattern_ok:
            fmt_bonus += 0.1

        total = min(slot_mean + fmt_bonus, 1.0)
        scores.append(total)

    device = None
    if embedding_layer is not None:
        device = embedding_layer.weight.device
    elif torch.cuda.is_available():
        device = "cuda"
    return torch.tensor(scores, dtype=torch.float32, device=device)


def compose_reward(
    pixel_r: torch.Tensor,
    text_r: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    w_pixel = float(cfg.get("pixel", 1.0))
    w_text = float(cfg.get("text", 1.0))
    return w_pixel * pixel_r + w_text * text_r
