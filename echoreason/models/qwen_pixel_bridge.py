# -*- coding: utf-8 -*-
"""
Qwen 与像素模块桥接：
  - L→M：将 <SEG#k> 对应的文本隐状态投射为掩膜查询
  - M→L：掩膜特征池化为 evidence tokens，插入 <EVIDENCE> 段
升级：掩膜池化时同时提取核心区与膨胀环（环境区）两类证据。
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenPixelBridge(nn.Module):
    def __init__(
        self,
        text_hidden_dim: int,
        query_dim: int,
        seg_token_id: int,
        evidence_token_dim: int,
        max_masks: int = 5,
        ring_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.seg_token_id = int(seg_token_id)
        self.max_masks = int(max_masks)
        self.ring_kernel = int(ring_kernel) if ring_kernel % 2 == 1 else int(ring_kernel + 1)  # ensure odd

        self.seg_to_query = nn.Linear(text_hidden_dim, query_dim)
        self.query_gate = nn.Linear(text_hidden_dim, 1)
        self.evidence_proj = nn.Linear(evidence_token_dim, text_hidden_dim)

    # ---------------- L -> M ---------------- #
    def _select_seg_hidden(
        self,
        text_hidden: torch.Tensor,   # [B,L,Dt]
        input_ids: torch.Tensor,     # [B,L]
    ) -> torch.Tensor:
        """
        取出 <SEG#1> 对应的文本隐状态；若缺失，则退化为最后一位。
        """
        seg_mask = input_ids.eq(self.seg_token_id)  # [B,L]
        if not seg_mask.any():
            return text_hidden[:, -1, :]            # [B,Dt]

        seg_hidden = []
        for b in range(text_hidden.size(0)):
            idx = torch.nonzero(seg_mask[b], as_tuple=False)
            if idx.numel() == 0:
                seg_hidden.append(text_hidden[b, -1, :])
            else:
                seg_hidden.append(text_hidden[b, idx[0, 0], :])
        return torch.stack(seg_hidden, dim=0)       # [B,Dt]

    def language_to_queries(
        self,
        text_hidden: torch.Tensor,   # [B,L,Dt]
        input_ids: torch.Tensor,     # [B,L]
        base_queries: torch.Tensor,  # [K,Dq] learnable queries
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
          query_init: [B,K,Dq]，第一条 query 由 <SEG#1> 驱动，其余复制 base_queries；
          query_gate: [B,K]    ，基于 seg hidden 生成的门控。
        """
        B = text_hidden.size(0)
        K, Dq = base_queries.shape

        seg_hidden = self._select_seg_hidden(text_hidden, input_ids)       # [B,Dt]
        seg_query = self.seg_to_query(seg_hidden)                          # [B,Dq]

        query_init = base_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        query_init[:, 0, :] = seg_query

        gate = torch.sigmoid(self.query_gate(seg_hidden)).expand(B, K)     # [B,K]
        return query_init, gate

    def make_attn_bias(
        self,
        focus_map: torch.Tensor,                # [B,1,H,W]
        pixel_feats: Dict[int, torch.Tensor],   # {8,16,32}
        num_queries: int,
    ) -> Dict[int, torch.Tensor]:
        """
        将 focus_map 下采样到各尺度，作为 DeformableTransformer 的先验 bias。
        """
        bias = {}
        K = int(num_queries)
        for lv, feat in pixel_feats.items():
            target_size = feat.shape[-2:]
            fm = F.interpolate(focus_map, size=target_size, mode="bilinear", align_corners=False)
            bias[lv] = fm.expand(-1, K, -1, -1)  # [B,K,Hl,Wl]
        return bias

    # ---------------- M -> L ---------------- #
    def _weighted_pool(self, weights: torch.Tensor, pixel_feat: torch.Tensor) -> torch.Tensor:
        """
        weights: [m,H,W] or [H,W]; pixel_feat: [C,H,W]; returns [m,C]
        """
        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
        norm = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        weights = weights / norm
        return torch.einsum("khw,chw->kc", weights, pixel_feat)

    def masks_to_evidence(
        self,
        mask_logits: torch.Tensor,        # [B,K,H,W]
        pixel_feat: torch.Tensor,         # [B,C,H,W] 选用最高分辨率特征
        mask_scores: Optional[torch.Tensor] = None,  # [B,K]
        max_masks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        将掩膜池化为 evidence token 嵌入：[B, M, Dt]，M<=max_masks。
        输出 core（主掩膜）与 context（膨胀环）两类证据，按序拼接。
        """
        B, K, H, W = mask_logits.shape
        m = max_masks if max_masks is not None else self.max_masks
        m = min(m, K)

        probs = torch.sigmoid(mask_logits)
        if mask_scores is not None:
            topk = torch.topk(mask_scores, k=m, dim=1).indices  # [B,m]
        else:
            areas = probs.flatten(2).sum(-1)  # [B,K]
            topk = torch.topk(areas, k=m, dim=1).indices

        ev_list = []
        pad = self.ring_kernel // 2
        for b in range(B):
            idx = topk[b]
            selected = probs[b, idx, :, :]                           # [m,H,W]
            core = self._weighted_pool(selected, pixel_feat[b])      # [m,C]

            dilated = F.max_pool2d(selected, kernel_size=self.ring_kernel, stride=1, padding=pad)
            ring_mask = (dilated - selected).clamp(min=0.0)
            empty = ring_mask.sum(dim=(1, 2), keepdim=True).eq(0)
            ring_mask = ring_mask + empty.float() * 1e-4
            context = self._weighted_pool(ring_mask, pixel_feat[b])  # [m,C]

            pooled = torch.cat([core, context], dim=0)  # [2m,C]
            ev = self.evidence_proj(pooled)             # [2m,Dt]
            ev_list.append(ev)

        max_m = max(ev.shape[0] for ev in ev_list)
        ev_padded = []
        for ev in ev_list:
            if ev.shape[0] < max_m:
                pad = ev.new_zeros(max_m - ev.shape[0], ev.shape[1])
                ev = torch.cat([ev, pad], dim=0)
            ev_padded.append(ev)
        evidence = torch.stack(ev_padded, dim=0)  # [B,max_m,Dt]
        return evidence

    def inject_evidence_tokens(
        self,
        input_ids: torch.Tensor,            # [B,L]
        attention_mask: torch.Tensor,       # [B,L]
        inputs_embeds: torch.Tensor,        # [B,L,Dt]
        evidence_embeds: torch.Tensor,      # [B,M,Dt]
        evidence_token_id: int,
        pad_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        将 evidence_embeds 插入到 <EVIDENCE> token 之后，返回新的 ids / mask / embeds。
        """
        B, L, Dt = inputs_embeds.shape
        M = evidence_embeds.size(1)
        device = input_ids.device

        new_ids_list = []
        new_mask_list = []
        new_embed_list = []
        evidence_ids = torch.full((M,), int(evidence_token_id), device=device, dtype=input_ids.dtype)

        for b in range(B):
            ids_b = input_ids[b]
            mask_b = attention_mask[b]
            embeds_b = inputs_embeds[b]
            evidence_b = evidence_embeds[b]

            pos = (ids_b == evidence_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                insert = L
            else:
                insert = int(pos[0, 0]) + 1  # 插在 <EVIDENCE> 之后

            new_ids = torch.cat([ids_b[:insert], evidence_ids, ids_b[insert:]], dim=0)
            new_mask = torch.cat(
                [mask_b[:insert], torch.ones(M, device=device, dtype=mask_b.dtype), mask_b[insert:]],
                dim=0,
            )
            new_embed = torch.cat([embeds_b[:insert], evidence_b, embeds_b[insert:]], dim=0)

            new_ids_list.append(new_ids)
            new_mask_list.append(new_mask)
            new_embed_list.append(new_embed)

        max_len = max(t.size(0) for t in new_ids_list)

        def _pad(t: torch.Tensor, value: float) -> torch.Tensor:
            pad_len = max_len - t.size(0)
            if pad_len == 0:
                return t
            pad_shape = (pad_len,) + t.shape[1:]
            pad_tensor = t.new_full(pad_shape, value)
            return torch.cat([t, pad_tensor], dim=0)

        ids_padded = []
        mask_padded = []
        embed_padded = []
        pad_id = pad_token_id if pad_token_id is not None else 0

        for ids, mask, emb in zip(new_ids_list, new_mask_list, new_embed_list):
            ids_padded.append(_pad(ids, float(pad_id)))
            mask_padded.append(_pad(mask, 0.0))
            embed_padded.append(_pad(emb, 0.0))

        new_input_ids = torch.stack(ids_padded, dim=0).to(dtype=input_ids.dtype)
        new_attention_mask = torch.stack(mask_padded, dim=0).to(dtype=attention_mask.dtype)
        new_inputs_embeds = torch.stack(embed_padded, dim=0)

        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "inputs_embeds": new_inputs_embeds,
        }
