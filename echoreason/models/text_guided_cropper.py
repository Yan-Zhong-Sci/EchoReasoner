# -*- coding: utf-8 -*-
"""
TextGuidedCropper: text-guided soft focus on visual features.
Pipeline:
  - Project text tokens and visual patches to a shared dim, compute similarity.
  - Build a focus_map over spatial positions from relevant text tokens.
  - Use focus_map as a soft gate to reweight visual features.
"""
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedCropper(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        focus_token_id: Optional[int] = None,
        topk_ratio: float = 0.2,
        background_scale: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            vision_dim: channels of visual features from the vision tower.
            text_dim: hidden dim of text tokens.
            hidden_dim: shared projection dim for similarity.
            focus_token_id: if set, only use this token for focus; default None uses all valid tokens.
            topk_ratio: keep top-k spatial responses, suppress others.
            background_scale: scaling factor for non-focus regions (0~1).
        """
        super().__init__()
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0,1].")

        self.focus_token_id = focus_token_id
        self.topk_ratio = float(topk_ratio)
        self.background_scale = float(background_scale)
        self.temperature = float(temperature)

        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def _build_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select tokens for focus:
          - if focus_token_id is set, only that id;
          - else all valid tokens (attention_mask==1).
        """
        if self.focus_token_id is not None:
            focus = input_ids.eq(self.focus_token_id) & attention_mask.bool()
        else:
            focus = attention_mask.bool()
        if not focus.any():
            focus = attention_mask.bool()
        return focus  # [B,L]

    def forward(
        self,
        vision_map: torch.Tensor,            # [B,Cv,H,W]
        text_hidden: Union[torch.Tensor, List[torch.Tensor]],  # [B,L,Dt] or list of last n layers
        input_ids: torch.Tensor,             # [B,L]
        attention_mask: torch.Tensor,        # [B,L]
    ) -> Dict[str, torch.Tensor]:
        B, Cv, H, W = vision_map.shape
        if isinstance(text_hidden, (list, tuple)):
            # fuse multi-layer text hidden (e.g., last 3 layers)
            stacked = torch.stack(text_hidden, dim=0)  # [n,B,L,D]
            text_fused = stacked.mean(dim=0)
        else:
            text_fused = text_hidden
        _, L, Dt = text_fused.shape

        if Dt != self.text_proj.in_features:
            raise ValueError(f"[TextGuidedCropper] text_dim {Dt} != init {self.text_proj.in_features}")
        if Cv != self.vision_proj.in_features:
            raise ValueError(f"[TextGuidedCropper] vision_dim {Cv} != init {self.vision_proj.in_features}")

        focus_mask = self._build_token_mask(input_ids, attention_mask)  # [B,L]
        token_weights = focus_mask.float() * attention_mask.float()
        token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        vision_flat = vision_map.flatten(2).transpose(1, 2)  # [B,HW,Cv]
        vision_proj = self.vision_proj(vision_flat)          # [B,HW,Dh]
        text_proj = self.text_proj(text_fused)              # [B,L,Dh]

        sim = torch.einsum("bld,bpd->blp", text_proj, vision_proj) * self.scale  # [B,L,HW]
        sim = sim.masked_fill(~focus_mask.unsqueeze(-1), -1e4)

        attn = F.softmax(sim / max(self.temperature, 1e-6), dim=-1)  # spatial softmax with temperature
        attn = attn * token_weights.unsqueeze(-1)  # [B,L,HW]

        focus_map = attn.sum(dim=1)  # [B,HW]
        focus_map = focus_map.view(B, 1, H, W)

        if self.topk_ratio < 1.0:
            k = max(int(H * W * self.topk_ratio), 1)
            thresh, _ = torch.topk(focus_map.view(B, -1), k=k, dim=-1)
            kth = thresh[:, -1].view(B, 1, 1, 1)
            mask_keep = focus_map >= kth
            focus_map = torch.where(mask_keep, focus_map, focus_map * self.background_scale)

        focus_map = focus_map / focus_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].clamp_min(1e-6)

        focused_feat = vision_map * (focus_map + self.background_scale * (1 - focus_map))

        return {
            "focus_map": focus_map,          # [B,1,H,W]
            "focused_feature": focused_feat, # [B,Cv,H,W]
        }
