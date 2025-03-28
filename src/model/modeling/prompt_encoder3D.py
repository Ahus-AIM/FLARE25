from typing import Optional, Tuple

import torch
from torch import nn

from .common import NormLayer3D
from .position_encoder3D import PositionEncoder3D


class DownscaleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
            NormLayer3D(),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiClickEmbedding(nn.Module):
    def __init__(self, embed_dim, num_points):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points

        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for i in range(self.num_points)])
        self.last_click_embedding = nn.Embedding(1, embed_dim)

    def get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        points = points + 0.5  # Shift to center of pixel
        point_embedding = torch.zeros((points.shape[0], points.shape[1], self.embed_dim), device=points.device)
        for i in range(self.num_points):
            point_embedding[labels == i] += self.point_embeddings[i].weight
        if labels.shape[1] > 0:
            point_embedding[:, -1] += self.last_click_embedding.weight
        return point_embedding


class BoxEmbedding(nn.Module):
    def __init__(self, embed_dim, num_points):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points

        self.box_embedding = nn.ModuleList([nn.Embedding(1, embed_dim) for i in range(self.num_points)])

    def get_device(self) -> torch.device:
        return self.box_embedding[0].weight.device

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        assert boxes.shape[2] == 3, f"Expected boxes to have shape Bx2x3, got {boxes.shape}"
        assert boxes.shape[1] == 2, f"Expected boxes to have shape Bx2x3, got {boxes.shape}"

        boxes = boxes + 0.5  # Shift to center of pixel
        corner_embedding = torch.zeros((boxes.shape[0], boxes.shape[1], self.embed_dim), device=boxes.device)
        corner_embedding[:, 0, :] += self.box_embedding[0].weight
        corner_embedding[:, 1, :] += self.box_embedding[1].weight
        return corner_embedding


class PromptEncoder3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        prev_mask_downscaling_factor: int,
        init_filters: int,
        position_encoder: PositionEncoder3D,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.position_encoder = position_encoder
        self.prev_mask_downscaling_factor = prev_mask_downscaling_factor

        self.num_point_embeddings: int = 2  # pos/neg point
        self.point_embeddings = MultiClickEmbedding(embed_dim, self.num_point_embeddings)

        self.num_corner_embeddings: int = 2  # box corners
        self.corner_embeddings = BoxEmbedding(embed_dim, self.num_corner_embeddings)

        downscale_layers = torch.log2(torch.tensor(prev_mask_downscaling_factor)).int().item()

        self.mask_downscaling = []
        num_channels = [1] + [init_filters * 2**i for i in range(downscale_layers)]
        for i in range(downscale_layers):
            self.mask_downscaling.append(DownscaleBlock3D(num_channels[i], num_channels[i + 1]))
        self.mask_downscaling.append(nn.Conv3d(num_channels[-1], embed_dim, kernel_size=1))
        self.mask_downscaling = nn.Sequential(*self.mask_downscaling)

        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe_term(self, batch_size: int, spatial_dims: tuple[int, int, int]) -> Optional[torch.Tensor]:
        return self.position_encoder.compute_dense_pe_term(batch_size, spatial_dims)  # 1xXxYxZ

    def get_dense_pe_factor(self, batch_size: int, spatial_dims: tuple[int, int, int]) -> Optional[torch.Tensor]:
        return self.position_encoder.compute_dense_pe_factor(batch_size, spatial_dims)  # 1xXxYxZ

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings.get_device()

    def forward(
        self,
        spatial_dims: tuple[int, int, int],
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        input_spatial_dims = tuple(s * self.prev_mask_downscaling_factor for s in spatial_dims)
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        sparse_embeddings_pe_term = None
        sparse_embeddings_pe_factor = None

        point_embeddings = None
        coords, labels = None, None
        if points is not None:
            coords, labels = points
            point_embeddings = self.point_embeddings(coords, labels)

        box_embeddings = self.corner_embeddings(boxes) if boxes is not None else None

        for click_object, embeddings, coords in zip(
            [boxes, points], [point_embeddings, box_embeddings], [coords, boxes]
        ):
            if embeddings is None:
                continue

            sparse_embeddings = torch.cat([sparse_embeddings, embeddings], dim=1)

            pe_term = self.position_encoder.compute_point_pe_term(coords, input_spatial_dims)
            if pe_term is not None:
                sparse_embeddings_pe_term = (
                    torch.cat([sparse_embeddings_pe_term, pe_term], dim=1)
                    if sparse_embeddings_pe_term is not None
                    else pe_term
                )

            pe_factor = self.position_encoder.compute_point_pe_factor(coords, input_spatial_dims)
            if pe_factor is not None:
                sparse_embeddings_pe_factor = (
                    torch.cat([sparse_embeddings_pe_factor, pe_factor], dim=1)
                    if sparse_embeddings_pe_factor is not None
                    else pe_factor
                )

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = torch.tensor(0.0, device=self._get_device())

        return sparse_embeddings, sparse_embeddings_pe_term, sparse_embeddings_pe_factor, dense_embeddings
