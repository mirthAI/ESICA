import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn
from torchtune.modules import RotaryPositionalEmbeddings


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, activation: Type[nn.Module]):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        head_dim = embedding_dim // num_heads
        self.rope = RotaryPositionalEmbeddings(dim=head_dim)

        for _ in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    rope=self.rope,
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads)
        self.norm_final_attn = nn.RMSNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        prompt_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        queries = prompt_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
            )

        attn_out = self.final_attn_token_to_image(
            q=queries,
            k=keys,
            v=keys,
            rope=self.rope,
        )
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        rope: RotaryPositionalEmbeddings,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.rope = rope

        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.RMSNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads)
        self.norm2 = nn.RMSNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.RMSNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads)
        self.norm4 = nn.RMSNorm(embedding_dim)

    def forward(self, queries: Tensor, keys: Tensor) -> Tuple[Tensor, Tensor]:
        attn_out = self.self_attn(
            q=queries,
            k=queries,
            v=queries,
            rope=self.rope,
        )
        queries = queries + attn_out
        queries = self.norm1(queries)

        attn_out = self.cross_attn_token_to_image(
            q=queries, k=keys, v=keys, rope=self.rope
        )
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        attn_out = self.cross_attn_image_to_token(
            q=keys, k=queries, v=queries, rope=self.rope
        )
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, seq_len, num_heads * head_dim)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rope: RotaryPositionalEmbeddings,
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q_heads = self._separate_heads(q)
        k_heads = self._separate_heads(k)
        v_heads = self._separate_heads(v)

        q_heads = q_heads.permute(0, 2, 1, 3)
        k_heads = k_heads.permute(0, 2, 1, 3)

        q_heads = rope(q_heads)
        k_heads = rope(k_heads)

        q_heads = q_heads.permute(0, 2, 1, 3)
        k_heads = k_heads.permute(0, 2, 1, 3)

        attn = q_heads @ k_heads.transpose(-2, -1)
        attn = attn / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v_heads
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
