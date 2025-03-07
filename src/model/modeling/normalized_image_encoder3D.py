from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


def normalize(x: Tensor, dim: int = -1) -> Tensor:
    res: Tensor = x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-3)
    return res


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        act: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lin_up = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.lin_down = nn.Linear(2 * embedding_dim, embedding_dim)
        self.act = act()

        self.mlp_alpha_init_value: float = 0.1
        self.mlp_alpha_init_scaling: float = 1.0
        self.mlp_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(embedding_dim, dtype=torch.float32)
        )

        self.suv_init_value: float = 1.0
        self.suv_init_scaling: float = 1.0
        self.suv: torch.nn.Parameter = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(4 * embedding_dim, dtype=torch.float32)
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
        # print("Normalizing MLP block")
        self.lin_up.weight.data.copy_(normalize(self.lin_up.weight.data, dim=1))
        self.lin_down.weight.data.copy_(normalize(self.lin_down.weight.data, dim=0))


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class NormalizedImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.SiLU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    img_size // patch_size,
                    img_size // patch_size,
                    img_size // patch_size,
                    embed_dim,
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
                input_size=(
                    img_size // patch_size,
                    img_size // patch_size,
                    img_size // patch_size,
                ),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = normalize(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 4, 1, 2, 3))

        x = normalize(x, dim=-1)

        return x

    def normalize_weights(self):
        self.pos_embed.data.copy_(normalize(self.pos_embed.data, dim=-1))


class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        act_layer: Type[nn.Module] = nn.SiLU,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=input_size,
        )

        self.mlp = MLPBlock(embedding_dim=dim, act=act_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x, residual_stream=x)
        x = self.mlp(x, residual_stream=x)
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.internal_dim = dim
        self.num_heads = num_heads
        self.softmax_scale = (dim // num_heads) ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_alpha_init_value: float = 0.1
        self.attn_alpha_init_scaling: float = 1.0
        self.attn_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(dim, dtype=torch.float32)
        )

        self.sqk_init_value: float = 1.0
        self.sqk_init_scaling: float = 1.0
        self.sqk: torch.nn.Parameter = torch.nn.Parameter(
            self.sqk_init_scaling * torch.ones(self.internal_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor, residual_stream: Optional[Tensor] = None) -> Tensor:
        B, D, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        sqk = (
            (self.sqk * (self.sqk_init_value / self.sqk_init_scaling))
            .view(self.num_heads, 1, self.internal_dim // self.num_heads)
            .repeat(B, 1, 1)
        )
        q = sqk * normalize(q, dim=-1)
        k = sqk * normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)

        attn = torch.softmax(self.softmax_scale * attn, dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        attn_out = self.proj(x)

        attn_out = normalize(attn_out, dim=-1)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        out = normalize(residual_stream + lr * (attn_out - residual_stream), dim=-1)

        return out

    def normalize_weights(self):
        self.qkv.weight.data.copy_(normalize(self.qkv.weight.data, dim=1))
        self.proj.weight.data.copy_(normalize(self.proj.weight.data, dim=0))


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16, 16),
        stride: Tuple[int, int] = (16, 16, 16),
        padding: Tuple[int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x
