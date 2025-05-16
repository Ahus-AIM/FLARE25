from abc import ABC
from typing import Any, Optional, Tuple

import torch
from torch import nn


class PositionEncoder3D(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, position: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def compute_dense_pe_term(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        raise NotImplementedError

    def compute_dense_pe_factor(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        raise NotImplementedError

    def compute_point_pe_term(self, position: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        raise NotImplementedError

    def compute_point_pe_factor(self, position: torch.Tensor) -> None:
        raise NotImplementedError


class PositionEmbeddingRandom3D(PositionEncoder3D):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, embed_dim: int, num_heads: Optional[int] = None) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((3, embed_dim // 2)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        if not (torch.all(coords >= 0) and torch.all(coords <= 1)):
            print(f"WARNING: Position should be normalized to [0, 1]. Got {coords}.")
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        pe_encoding = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return pe_encoding

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        x, y, z = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((x, y, z), device=device, dtype=torch.float32)
        x_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        x_embed = x_embed / x
        y_embed = y_embed / y
        z_embed = z_embed / z

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x X x Y x Z

    def _forward_with_coords(self, coords_input: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / spatial_dims[0]
        coords[:, :, 1] = coords[:, :, 1] / spatial_dims[1]
        coords[:, :, 2] = coords[:, :, 2] / spatial_dims[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def compute_dense_pe_term(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        pe = self(spatial_dims)
        return pe.unsqueeze(0).repeat(
            batch_size,
            *[1] * len(pe.shape),
        )  # batch x seq x embed_dim x embed_dim

    def compute_dense_pe_factor(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> None:
        return None

    def compute_point_pe_term(self, position: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        return self._forward_with_coords(position, spatial_dims)

    def compute_point_pe_factor(self, position: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> None:
        return None


class SkewSymmetricPositionEncoder(PositionEncoder3D):
    def __init__(self, embed_dim: int, block_size: int, num_dims: int = 3):
        super().__init__()
        # TODO: Implement support for uniqe matrices for each head.
        # TODO: Implement support for each attention block.
        self.embed_dim = embed_dim
        self.num_dims = num_dims
        self.block_size = block_size

        num_blocks = embed_dim // block_size

        self.parameter = nn.Parameter(
            torch.rand(
                self.num_dims,
                num_blocks,
                self.block_size,
                self.block_size,
            )
            * 2
            * torch.pi
        )

    def calculate_skew_symmetric_matrix(self, parameter):
        p_upper = torch.triu(parameter, diagonal=1)
        p_skew_symmetric = p_upper - torch.transpose(p_upper, -1, -2)
        skew_symmetric_matrix = torch.stack(
            [
                torch.block_diag(*[p_skew_symmetric[i, j] for j in range(p_skew_symmetric.shape[1])])
                for i in range(p_skew_symmetric.shape[0])
            ]
        )
        return skew_symmetric_matrix

    def _compute_rotary_matrix(self, position: torch.Tensor) -> torch.Tensor:
        if not (torch.all(position >= 0) and torch.all(position <= 1)):
            print(f"WARNING: Position should be normalized to [0, 1]. Got {position}.")
        skew_symmetric_matrix = self.calculate_skew_symmetric_matrix(self.parameter).clone()
        pos_sum = (position.unsqueeze(-1).unsqueeze(-1) * skew_symmetric_matrix).sum(dim=-3)
        rot_mat = torch.matrix_exp(pos_sum)
        return rot_mat  # batch x num_points x embed_dim x embed_dim

    def compute_dense_pe_term(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> None:
        return None

    def compute_dense_pe_factor(self, batch_size: int, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        coordinates = (
            torch.cartesian_prod(*[torch.arange(s, device=self.parameter.device) for s in spatial_dims]) + 0.5
        ) / max(spatial_dims)

        rot_mat = self._compute_rotary_matrix(coordinates)
        return rot_mat.unsqueeze(0).repeat(
            batch_size,
            *[1] * len(rot_mat.shape),
        )  # batch x seq x embed_dim x embed_dim

    def compute_point_pe_term(self, position: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> None:
        return None

    def compute_point_pe_factor(self, position: torch.Tensor, spatial_dims: Tuple[int, int, int]) -> torch.Tensor:
        position = (position + 0.5) / max(spatial_dims)
        return self._compute_rotary_matrix(position)


class LieRE(SkewSymmetricPositionEncoder):
    def __init__(self, embed_dim: int, num_heads: int, num_dims: int = 3):
        super().__init__(embed_dim // num_heads, embed_dim // num_heads, num_dims)


class RoPEMixed(SkewSymmetricPositionEncoder):
    def __init__(self, embed_dim: int, num_heads: int, num_dims: int = 3):
        super().__init__(embed_dim // num_heads, 2, num_dims)


if __name__ == "__main__":
    # Boxes: [batch, num boxes, coords(3)]
    # Points: [batch, num points, coords(3)]

    embed_dim = 64
    num_heads = 1
    num_dims = 3

    num_points = 10
    num_batches = 6

    s = LieRE(embed_dim=embed_dim, num_heads=num_heads, num_dims=num_dims)
    s = RoPEMixed(embed_dim=embed_dim, num_heads=num_heads, num_dims=num_dims)
    p = torch.randint(0, 10, (num_batches, num_points, num_dims))
    points_emb = torch.rand((num_batches, num_points, embed_dim))

    s(points_emb, p)
