# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from .dcformer import DecompConv3D
from .module import RMSNorm3D


class SkipUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="trilinear", align_corners=False
            ),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size // 2),
            ),
            # RMSNorm3D(out_channels),
            # nn.GELU(),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels + out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            RMSNorm3D(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        upsampled = self.up_conv(x)
        res = upsampled

        if skip is not None:
            upsampled = torch.cat((upsampled, skip), dim=1)

        return self.conv(upsampled) + res


class ImageDecoder(nn.Module):
    def __init__(
        self,
        high_res_dim: List[int] = [48, 96, 192, 384, 768],
        kernel_size: List[int] = [13, 11, 9, 7, 5],
        hidden_dim: int = 48,
        up_scale: int = 2,
    ) -> None:
        super().__init__()
        self.high_res_dim = high_res_dim
        self.num_blocks = len(high_res_dim)

        self.adapters = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if high_res_dim[i] == hidden_dim
                    else DecompConv3D(
                        in_dim=high_res_dim[i],
                        out_dim=hidden_dim,
                        kernel_size=kernel_size[i],
                        stride=1,
                        act=nn.GELU(),
                        fd=True,
                    )
                )
                for i in range(self.num_blocks)
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                SkipUpBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    scale_factor=up_scale,
                )
                for _ in range(self.num_blocks - 1)
            ]
        )

    def forward(self, src: torch.Tensor, high_res: List[torch.Tensor]) -> torch.Tensor:
        x = high_res + [src]

        for i in range(self.num_blocks):
            x[i] = self.adapters[i](x[i])

        x.reverse()

        upsampled = x[0]
        for i in range(self.num_blocks - 1):
            upsampled = self.up_blocks[i](upsampled, x[i + 1])

        return upsampled


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        text_dim: int,
        transformer: nn.Module,
        feat_shape: Tuple[int, int, int],
        activation: Type[nn.Module] = nn.GELU,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.feat_shape = feat_shape

        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.upscaling = ImageDecoder(hidden_dim=hidden_dim)

        self.txt_align_upscaled_embedding = nn.Sequential(
            nn.Linear(text_dim, transformer_dim // 4),
            nn.RMSNorm(transformer_dim // 4),
            activation(),
            nn.Linear(transformer_dim // 4, transformer_dim // 8),
            nn.RMSNorm(transformer_dim // 8),
            activation(),
            nn.Linear(transformer_dim // 8, hidden_dim),
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: List[torch.Tensor],
    ) -> torch.Tensor:
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embedding
        src = src + dense_prompt_embeddings

        b, c, h, w, d = src.shape

        hs, src = self.transformer(src, tokens)
        mask_tokens_out = hs[:, :1, :]
        src = src.transpose(1, 2).view(b, c, h, w, d)

        upscaled_embedding = self.upscaling(src, high_res_features)

        b, c, h, w, d = upscaled_embedding.shape

        text_embedding_down = self.txt_align_upscaled_embedding(
            text_embedding
        ).unsqueeze(dim=1)
        upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
        mask = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)

        return mask, mask_tokens_out
