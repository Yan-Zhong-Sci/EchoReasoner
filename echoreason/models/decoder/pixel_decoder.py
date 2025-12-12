"""
Pixel decoder: refine multi-scale features and provide flattening utilities.
Assumes input pyramid contains 1/4, 1/8, 1/16, 1/32 features (keys 4,8,16,32).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layer(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p),
        nn.GroupNorm(32, out_c),
        nn.ReLU(inplace=True),
    )


class PixelDecoder(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int) -> None:
        super().__init__()
        self.scales = [4, 8, 16, 32]
        self.refine = nn.ModuleDict(
            {str(s): _make_layer(in_channels, feat_channels, k=3, p=1) for s in self.scales}
        )

    def forward(self, pyramid: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Refine each scale with a GN+ReLU conv. If某尺度缺失，则从最接近的尺度插值获得。
        """
        out: Dict[int, torch.Tensor] = {}
        available = sorted(pyramid.keys())
        for s in self.scales:
            if s in pyramid:
                feat = pyramid[s]
            else:
                # 从最近的已有尺度插值
                nearest = min(available, key=lambda k: abs(k - s))
                feat = F.interpolate(
                    pyramid[nearest],
                    scale_factor=float(s) / float(nearest),
                    mode="bilinear",
                    align_corners=False,
                )
            out[s] = self.refine[str(s)](feat)
        return out

    @staticmethod
    def _get_position_embedding(tensor: torch.Tensor, num_pos_feats: int) -> torch.Tensor:
        """
        正弦位置编码，形状 [B, C, H, W]，与输入特征同尺度同通道。
        """
        mask = torch.zeros((tensor.shape[0], tensor.shape[2], tensor.shape[3]), device=tensor.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * torch.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    @staticmethod
    def flatten_for_transformer(pixels: Dict[int, torch.Tensor]):
        """
        Flattens multi-scale features for transformer cross-attention.
        Returns:
            value: [B, sum(H_l*W_l), C]
            spatial_shapes: [L, 2] tensor of (H, W) per level
            level_start_index: [L] tensor of start offsets per level
            pos_embed: [B, sum(H_l*W_l), C] flatten 位置编码
        """
        levels = sorted(pixels.keys())
        values = []
        spatial_shapes = []
        level_start_index = []
        pos_embeds = []
        offset = 0
        for lv in levels:
            feat = pixels[lv]
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            level_start_index.append(offset)
            values.append(feat.flatten(2).transpose(1, 2))  # [B, HW, C]
            pos = PixelDecoder._get_position_embedding(feat, C // 2)
            pos_embeds.append(pos.flatten(2).transpose(1, 2))
            offset += H * W

        value = torch.cat(values, dim=1)  # [B, S, C]
        spatial_shapes_tensor = torch.tensor(spatial_shapes, device=value.device, dtype=torch.long)
        level_start_index_tensor = torch.tensor(level_start_index, device=value.device, dtype=torch.long)
        pos_embed = torch.cat(pos_embeds, dim=1)  # [B, S, C]
        return value, spatial_shapes_tensor, level_start_index_tensor, pos_embed
