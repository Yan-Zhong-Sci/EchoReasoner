"""
Mask2Former-style segmentation head used in ECHOReasoner.
FPN fusion with GroupNorm: multi-layer visual features -> 4/8/16/32 pyramid -> transformer decoder -> mask head.
"""
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import DeformableTransformerDecoder, MaskHead, PixelDecoder


def _make_layer(in_c: int, out_c: int, k: int, p: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p),
        nn.GroupNorm(32, out_c),
        nn.ReLU(inplace=True),
    )


class SimpleFPN(nn.Module):
    """Simple FPN: lateral 1x1 + top-down fusion + 3x3 smoothing (with GN+ReLU)."""

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 4) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList([_make_layer(in_channels, out_channels, 1) for _ in range(num_levels)])
        self.fpn_convs = nn.ModuleList([_make_layer(out_channels, out_channels, 3, 1) for _ in range(num_levels)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: list[tensor] from shallow -> deep
        laterals = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        outs = [None] * len(laterals)
        # top-down
        outs[-1] = self.fpn_convs[-1](laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            top = F.interpolate(outs[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            merged = laterals[i] + top
            outs[i] = self.fpn_convs[i](merged)
        return outs


class Mask2FormerSegmentation(nn.Module):
    def __init__(
        self,
        vision_channels: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        pretrained_path: str = None,
        pretrained_strict: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)

        # adapters for pyramid
        self.adapter4 = _make_layer(vision_channels, hidden_dim, 1)
        self.adapter8 = _make_layer(vision_channels, hidden_dim, 1)
        self.adapter16 = _make_layer(vision_channels, hidden_dim, 1)
        self.adapter32 = _make_layer(vision_channels, hidden_dim, 1)

        self.fpn = SimpleFPN(in_channels=vision_channels, out_channels=hidden_dim, num_levels=4)

        self.pixel_decoder = PixelDecoder(in_channels=hidden_dim, feat_channels=hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_decoder = DeformableTransformerDecoder(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            num_points=num_points,
            dropout=dropout,
        )
        self.mask_head = MaskHead(in_channels=hidden_dim, hidden_dim=hidden_dim)

        if pretrained_path:
            self._load_pretrained(pretrained_path, strict=pretrained_strict)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load_pretrained(self, path: str, strict: bool = False) -> None:
        import os
        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"[Mask2FormerSegmentation] pretrained_path not found: {path}")
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=strict)

    def _adapt(self, pyramid: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        # ensure 4/8/16/32 exist; interpolate from nearest if missing
        scales = sorted(pyramid.keys())
        adapted = {}

        def get_feat(scale: int) -> torch.Tensor:
            if scale in pyramid:
                return pyramid[scale]
            nearest = min(scales, key=lambda s: abs(s - scale))
            return F.interpolate(pyramid[nearest], scale_factor=scale / nearest, mode="bilinear", align_corners=False)

        p4 = get_feat(4)
        p8 = get_feat(8)
        p16 = get_feat(16)
        p32 = get_feat(32)

        adapted[4] = self.adapter4(p4)
        adapted[8] = self.adapter8(p8)
        adapted[16] = self.adapter16(p16)
        adapted[32] = self.adapter32(p32)
        return adapted

    def _fpn_build(self, feats: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        outs = self.fpn(feats)  # list same resolution as inputs
        p_high = outs[0]  # shallow fused
        p4 = F.interpolate(p_high, scale_factor=2, mode="bilinear", align_corners=False)
        p8 = p_high
        p16 = F.avg_pool2d(outs[min(1, len(outs) - 1)], kernel_size=2, stride=2)
        p32 = F.avg_pool2d(outs[min(2, len(outs) - 1)], kernel_size=4, stride=4)
        return {4: p4, 8: p8, 16: p16, 32: p32}

    def forward(
        self,
        features: Union[Dict[int, torch.Tensor], List[torch.Tensor], torch.Tensor],
        query_init: torch.Tensor = None,           # [B,K,C] optional language-guided init
        attn_bias: Dict[int, torch.Tensor] = None, # {scale: [B,K,H_l,W_l]}
        query_gate: torch.Tensor = None,           # [B,K]
    ) -> Dict[str, Any]:
        if isinstance(features, dict):
            pyramid = self._adapt(features)
        elif isinstance(features, (list, tuple)):
            pyramid = self._fpn_build(features)
        else:  # single feature
            pyramid = self._fpn_build([features])

        pixel_feats = self.pixel_decoder(pyramid)         # {4,8,16,32}: [B,C,H,W]
        value, spatial_shapes, level_start_index, pos_embed = self.pixel_decoder.flatten_for_transformer(
            pixel_feats
        )

        B = value.size(0)
        if query_init is None:
            query_init = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        if query_gate is None:
            query_gate = query_init.new_ones(B, query_init.size(1))

        dec_out = self.transformer_decoder(
            query=query_init,
            value=value,
            pos_embed=pos_embed,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            attn_bias=attn_bias,
            query_gate=query_gate,
        )

        mask_out = self.mask_head(
            pixel_feats=pixel_feats,
            query_feats=dec_out["query_feats"],
            aux_query_feats=dec_out["aux_query_feats"],
        )

        return {
            "pixel_feats": pixel_feats,
            "decoder": dec_out,
            "masks": mask_out["masks"],           # [B,K,H,W]
            "scores": mask_out["scores"],         # [B,K]
            "quality": mask_out["quality"],       # [B,K]
            "logits": mask_out["logits"],         # [B,K,H,W]
            "aux": mask_out["aux"],
        }
