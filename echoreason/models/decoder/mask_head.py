"""
Mask prediction head: projects transformer queries to per-pixel mask logits and scores.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj_feat = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.proj_query = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def _predict(
        self,
        pixel_feats: torch.Tensor,  # [B,C,H,W]
        query_feats: torch.Tensor,  # [B,K,C]
    ) -> Dict[str, torch.Tensor]:
        feat = self.proj_feat(pixel_feats)  # [B,C,H,W]
        q = self.proj_query(query_feats)    # [B,K,C]
        masks = torch.einsum("bkc,bchw->bkhw", q, feat)  # [B,K,H,W]
        scores = self.score_head(query_feats).squeeze(-1)  # [B,K]
        quality = torch.sigmoid(self.quality_head(query_feats).squeeze(-1))  # [B,K]
        return {
            "logits": masks,
            "masks": torch.sigmoid(masks),
            "scores": scores,
            "quality": quality,
        }

    def forward(
        self,
        pixel_feats: Dict[int, torch.Tensor],   # expects highest resolution at key 8
        query_feats: torch.Tensor,              # [B,K,C]
        aux_query_feats: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if 8 not in pixel_feats:
            raise KeyError("[MaskHead] pixel_feats must contain level 8.")

        main = self._predict(pixel_feats[8], query_feats)
        aux_outputs = []
        if aux_query_feats:
            for q_aux in aux_query_feats:
                aux_outputs.append(self._predict(pixel_feats[8], q_aux))

        main["aux"] = aux_outputs
        return main
