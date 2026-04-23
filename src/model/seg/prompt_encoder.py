from typing import Any, Optional, Tuple, Type

import torch
import torch.nn as nn

from .dcformer import DecompConv3D


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        self.mask_downscaling = nn.Sequential(
            DecompConv3D(
                1, embed_dim // 8, kernel_size=13, stride=2, act=activation(), fd=True
            ),
            DecompConv3D(
                embed_dim // 8,
                embed_dim // 4,
                kernel_size=11,
                stride=2,
                act=activation(),
                fd=True,
            ),
            DecompConv3D(
                embed_dim // 4,
                embed_dim // 2,
                kernel_size=7,
                stride=2,
                act=activation(),
                fd=True,
            ),
            DecompConv3D(
                embed_dim // 2,
                embed_dim,
                kernel_size=5,
                stride=2,
                act=activation(),
                fd=True,
            ),
        )

        self.no_mask_embed = nn.Parameter(
            torch.randn(1, embed_dim, *image_embedding_size)
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sam_tokens_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = text_embedding.shape[0]

        sparse_embeddings = text_embedding.unsqueeze(dim=1)

        if sam_tokens_out is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, sam_tokens_out], dim=1)

        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.expand(bs, -1, -1, -1, -1)

        return sparse_embeddings, dense_embeddings
