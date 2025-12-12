"""
Minimal transformer decoder for Mask2Former-style query-to-pixel attention.
Uses multi-head cross-attention over flattened pixel features with optional
attention bias derived from text-guided focus maps.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,              # [B, K, C]
        value: torch.Tensor,              # [B, S, C]
        pos_embed: Optional[torch.Tensor] = None,  # [B, S, C]
        attn_bias: Optional[torch.Tensor] = None,  # [B, K, S]
        query_gate: Optional[torch.Tensor] = None, # [B, K]
    ) -> torch.Tensor:
        B, K, _ = query.shape
        S = value.size(1)

        # projections
        q = self.q_proj(query)
        key_src = value if pos_embed is None else (value + pos_embed)
        k = self.k_proj(key_src)
        v = self.v_proj(value)

        if query_gate is not None:
            q = q * query_gate.unsqueeze(-1)

        # [B, H, L, D]
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, K, S]
        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(1)  # broadcast over heads

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, K, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, K, self.embed_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = self.norm1(out + query)
        out_ffn = self.ffn(out)
        out = self.norm2(out + out_ffn)
        return out


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_points: int = 4,  # kept for API compatibility
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    @staticmethod
    def _flatten_bias(attn_bias: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Flattens bias dict {scale: [B,K,H,W]} to [B,K,S] matching flattened pixels.
        Assumes scales are provided in ascending order (8,16,32).
        """
        if not attn_bias:
            return None
        levels = sorted(attn_bias.keys())
        flat = []
        for lv in levels:
            bias_lv = attn_bias[lv]
            B, K, H, W = bias_lv.shape
            flat.append(bias_lv.view(B, K, H * W))
        return torch.cat(flat, dim=-1)  # [B, K, S]

    def forward(
        self,
        query: torch.Tensor,                     # [B, K, C]
        value: torch.Tensor,                     # [B, S, C]
        pos_embed: Optional[torch.Tensor] = None, # [B, S, C]
        spatial_shapes: Optional[torch.Tensor] = None,   # [L, 2], kept for API compatibility
        level_start_index: Optional[torch.Tensor] = None,# [L], kept for API compatibility
        attn_bias: Optional[Dict[int, torch.Tensor]] = None,
        query_gate: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        bias_flat = self._flatten_bias(attn_bias) if attn_bias is not None else None
        q = query
        aux: List[torch.Tensor] = []
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            # 将原始 value 与 pos_embed 一并传入，Key 内部加位置，Value 保持纯净
            q = layer(
                query=q,
                value=value,
                pos_embed=pos_embed,
                attn_bias=bias_flat,
                query_gate=query_gate,
            )
            if i != last:
                aux.append(q)

        return {
            "query_feats": q,
            "aux_query_feats": aux,
        }
