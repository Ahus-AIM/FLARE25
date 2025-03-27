from __future__ import annotations

import math
from collections.abc import Callable
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Conv
from monai.utils import UpsampleMode
from torch import Tensor

from .common import SegResBlock, aniso_kernel, normalize


class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lin_up = nn.Linear(embedding_dim, 2 * mlp_dim)
        self.lin_down = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

        self.mlp_alpha_init_value: float = 0.1
        self.mlp_alpha_init_scaling: float = 1.0
        self.mlp_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(embedding_dim, dtype=torch.float32)
        )

        self.suv_init_value: float = 1.0
        self.suv_init_scaling: float = 1.0
        self.suv: torch.nn.Parameter = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * mlp_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor, residual_stream: Tensor) -> Tensor:
        # Apply MLP
        uv = self.lin_up(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.embedding_dim**0.5))
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        x_mlp = u * self.act(v)
        x_mlp = self.lin_down(x_mlp)

        # Step on the hypersphere
        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        residual_stream = normalize(residual_stream, dim=-1)
        x_mlp = normalize(x_mlp, dim=-1)

        out = normalize(residual_stream + lr * (x_mlp - residual_stream), dim=-1)

        return out

    def normalize_weights(self):
        self.lin_up.weight.data.copy_(normalize(self.lin_up.weight.data, dim=1))
        self.lin_down.weight.data.copy_(normalize(self.lin_down.weight.data, dim=0))


class NormalizedTwoWayTransformer3D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe_term: Optional[Tensor],
        image_pe_factor: Optional[Tensor],
        point_embedding: Tensor,
        point_pe_term: Optional[Tensor],
        point_pe_factor: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            point_embedding, image_embedding = layer(
                queries=point_embedding,
                keys=image_embedding,
                query_pe_term=point_pe_term,
                query_pe_factor=point_pe_factor,
                key_pe_term=image_pe_term,
                key_pe_factor=image_pe_factor,
            )

        return image_embedding


class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe_term: Optional[Tensor],
        query_pe_factor: Optional[Tensor],
        key_pe_term: Optional[Tensor],
        key_pe_factor: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if (query_pe_term is None) != (key_pe_term is None):
            raise ValueError("PE terms must be provided for both queries and keys or neither.")

        add_pe_terms = query_pe_term is not None

        queries_pe = queries
        if add_pe_terms:
            queries_pe = queries + query_pe_term

        queries = self.self_attn(
            q=queries_pe,
            k=queries_pe,
            v=queries,
            residual_stream=queries,
            query_pe=query_pe_factor,
            key_pe=query_pe_factor,
        )

        queries_pe = queries
        keys_pe = keys
        if add_pe_terms:
            queries_pe = queries + query_pe_term.reshape(queries.shape)
            keys_pe = keys + key_pe_term.reshape(keys.shape)

        queries = self.cross_attn_token_to_image(
            q=queries_pe, k=keys_pe, v=keys, residual_stream=queries, query_pe=query_pe_factor, key_pe=key_pe_factor
        )

        queries = self.mlp(queries, residual_stream=queries)

        queries_pe = queries
        if add_pe_terms:
            queries_pe = queries + query_pe_term.reshape(queries.shape)

        keys = self.cross_attn_image_to_token(
            q=keys_pe, k=queries_pe, v=queries, residual_stream=keys, query_pe=key_pe_factor, key_pe=query_pe_factor
        )

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.attn_alpha_init_value: float = 0.1
        self.attn_alpha_init_scaling: float = 1.0
        self.attn_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(embedding_dim, dtype=torch.float32)
        )

        self.sqk_init_value: float = 1.0
        self.sqk_init_scaling: float = 1.0
        self.sqk: torch.nn.Parameter = torch.nn.Parameter(
            self.sqk_init_scaling * torch.ones(self.internal_dim, dtype=torch.float32)
        )

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, residual_stream: Tensor | None, query_pe=None, key_pe=None
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.num_heads, 1, self.internal_dim // self.num_heads
        )

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        if query_pe is not None:
            q = (query_pe.unsqueeze(1) @ q.unsqueeze(-1)).squeeze(-1)
        if key_pe is not None:
            k = (key_pe.unsqueeze(1) @ k.unsqueeze(-1)).squeeze(-1)

        q = sqk * normalize(q, dim=-1)
        k = sqk * normalize(k, dim=-1)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn * math.sqrt(c_per_head)  # sqrt(d_k) instead of 1/sqrt(d_k) as q and k are normalized.
        attn = torch.softmax(attn, dim=-1)

        # Get output
        attn_out = attn @ v
        attn_out = self._recombine_heads(attn_out)
        attn_out = self.out_proj(attn_out)
        attn_out = normalize(attn_out, dim=-1)

        if residual_stream is not None:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            attn_out = normalize(residual_stream + lr * (attn_out - residual_stream), dim=-1)

        return attn_out

    def normalize_weights(self):
        self.q_proj.weight.data.copy_(normalize(self.q_proj.weight.data, dim=1))
        self.k_proj.weight.data.copy_(normalize(self.k_proj.weight.data, dim=1))
        self.v_proj.weight.data.copy_(normalize(self.v_proj.weight.data, dim=1))
        self.out_proj.weight.data.copy_(normalize(self.out_proj.weight.data, dim=0))


class NormalizedMaskDecoder3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        activation: Type[nn.Module],
        init_filters: int,
        spatial_dims: int = 3,
        norm: tuple | str = "batch",
        blocks_up: tuple = (4, 2, 2, 1),
        dsdepth: int = 1,
        preprocess: nn.Module | Callable | None = None,
        upsample_mode: UpsampleMode | str = "deconv",
        resolution: tuple | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.transformer = NormalizedTwoWayTransformer3D(
            depth=depth,
            embedding_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            activation=activation,
            num_heads=8,
            attention_downsample_rate=1,
        )
        kernel_size, padding, stride = aniso_kernel((2, 2, 2))
        self.dsdepth = 1
        n_up = len(blocks_up) - 1
        filters = init_filters * 2**n_up
        self.up_layers = nn.ModuleList()
        for i in range(n_up):
            filters = filters // 2
            level = nn.ModuleDict()
            level["upsample"] = UpSample(
                mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=2 * filters,
                out_channels=filters,
                kernel_size=kernel_size,
                scale_factor=stride,
                bias=False,
                align_corners=False,
            )
            blocks = [
                SegResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=filters,
                    kernel_size=kernel_size,
                    norm=norm,
                    act="relu",
                )
                for _ in range(blocks_up[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)
            level["head"] = Conv[Conv.CONV, spatial_dims](
                in_channels=filters,
                out_channels=1,
                kernel_size=1,
                bias=True,
            )
            self.up_layers.append(level)
        self.up_layers[-1]["head"].bias.data.fill_(-5)

    def output_upscaling(self, x, x_down):
        for i, level in enumerate((self.up_layers)):
            x = level["upsample"](x)
            x = x + x_down[i]
            x = level["blocks"](x)

        x = level["head"](x)
        return x

    def forward(
        self,
        step_wise_image_embeddings: List[torch.Tensor],
        image_pe_term: Optional[torch.Tensor],
        image_pe_factor: Optional[torch.Tensor],
        sparse_prompt_embeddings: Optional[torch.Tensor],
        sparse_prompt_embeddings_pe_term: Optional[torch.Tensor],
        sparse_prompt_embeddings_pe_factor: Optional[torch.Tensor],
        dense_prompt_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = step_wise_image_embeddings[0]
        b, c, x, y, z = image_embeddings.shape
        image_embeddings = image_embeddings + dense_prompt_embeddings

        image_embeddings = self.transformer(
            image_embeddings,
            image_pe_term,
            image_pe_factor,
            sparse_prompt_embeddings,
            sparse_prompt_embeddings_pe_term,
            sparse_prompt_embeddings_pe_factor,
        )

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(1, 2).view(b, c, x, y, z) * self.embed_dim**0.5
        mask_logits = self.output_upscaling(image_embeddings, step_wise_image_embeddings[1:])

        return mask_logits
