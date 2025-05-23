import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms.croppad.array import DivisiblePad

from src.custom_types import (
    BatchedImageLogits,
    BatchedPointCoords,
    Boxes,
    Image,
)


class CropOrPad(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor, seed):
        # ensure the input has shape target_shape in the last 3 dimensions
        random.seed(seed)
        input_shape = tensor.shape[-3:]
        if input_shape == self.target_shape:
            return tensor
        if input_shape[0] < self.target_shape[0]:
            padding = self.target_shape[0] - input_shape[0]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            tensor = F.pad(tensor, (0, 0, 0, 0, padding_left, padding_right))
        if input_shape[1] < self.target_shape[1]:
            padding = self.target_shape[1] - input_shape[1]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            tensor = F.pad(tensor, (0, 0, padding_left, padding_right, 0, 0))
        if input_shape[2] < self.target_shape[2]:
            padding = self.target_shape[2] - input_shape[2]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            tensor = F.pad(tensor, (padding_left, padding_right, 0, 0, 0, 0))

        return tensor


class Flip(torch.nn.Module):
    def __init__(self):
        """
        Randomly transpose the input tensor along one of the three possible axes.

        Additionally, randomly flip a dimension of the tensor.
        """
        super().__init__()
        self.transpose_dims = [[-3, -1, -2], [-1, -2, -3], [-2, -3, -1]]
        self.flip_dims = [-1, -2, -3]

    def forward(self, tensor, seed):
        random.seed(seed)
        if random.random() < 0.5:
            flip = random.choice(self.flip_dims)
            tensor = torch.flip(tensor, [flip])
        if random.random() < 0.5:
            tensor = tensor.permute(*random.choice(self.transpose_dims))
        return tensor


class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, tensor, seed):
        for transform in self.transforms:
            tensor = transform(tensor, seed)
        return tensor


class VolumeTransforms:
    def __init__(self, size_threshold: int) -> None:
        self.size_threshold: int = size_threshold
        self.pooling_factors: tuple[int, ...] = (1, 1, 1)
        self.crop_slices: tuple[slice, ...] | None = None  # set in forward and used in backward
        self.pad_values: tuple[int, ...] | None = None
        self.orig_shape: tuple[int, ...] | None = None
        self.cropped_shape: tuple[int, ...] | None = None
        self.padder = DivisiblePad(k=8, method="end", value=0)

    @staticmethod
    def _normalize_volume(volume: Image) -> Image:
        volume = volume.clone().float()
        volume[volume <= 0] = torch.nan
        positive_volume = volume[~torch.isnan(volume)]
        if positive_volume.numel() == 0:
            return torch.zeros_like(volume)
        min_val = positive_volume.min()
        max_val = positive_volume.max()
        volume = (volume - min_val + 1) / (max_val - min_val + 1)
        volume[torch.isnan(volume)] = 0
        return volume

    def pool_volume(
        self,
        volume: Image,
    ) -> Image:
        """Downsample the volume while also storing the pooling factors."""

        self.pooling_factors = (1, 1, 1)

        # F.max_pool3d expects channel dimension
        volume = volume.unsqueeze(0)

        while volume.numel() > self.size_threshold:
            min_dim = int(np.argmin(volume.shape[1:]))
            kernel_size = tuple(1 if i == min_dim else 2 for i in range(3))
            volume = F.max_pool3d(volume, kernel_size=kernel_size)
            self.pooling_factors = tuple(self.pooling_factors[i] * kernel_size[i] for i in range(3))
        # Remove channel dimension
        volume = volume.squeeze(0)

        return volume

    def pool_boxes(self, boxes: Boxes) -> Boxes:
        """
        Pool the boxes by dividing their coordinates by the pooling factors.
        """
        if boxes is None:
            return boxes
        return boxes / torch.tensor(self.pooling_factors, device=boxes.device).view(1, 3)

    def pool_coords(self, point_coords: BatchedPointCoords) -> BatchedPointCoords:
        """
        Pool the point coordinates by dividing them by the pooling factors.
        """
        return point_coords / torch.tensor(self.pooling_factors, device=point_coords.device).view(1, 3)

    def crop_point_coords(self, point_coords: BatchedPointCoords) -> BatchedPointCoords:
        """
        Crop the point coordinates by applying the crop slices.
        """
        assert self.crop_slices is not None, "crop_slices must be set before calling crop_point_coords"

        cropped_point_coords = point_coords.clone()
        cropped_point_coords[0][..., 0] = point_coords[0][..., 0] - self.crop_slices[0].start
        cropped_point_coords[0][..., 1] = point_coords[0][..., 1] - self.crop_slices[1].start
        cropped_point_coords[0][..., 2] = point_coords[0][..., 2] - self.crop_slices[2].start
        return cropped_point_coords

    def _crop(
        self, volume: Image, boxes: Boxes, point_coords: BatchedPointCoords | None
    ) -> tuple[Image, Boxes, BatchedPointCoords | None]:
        """Return cropped versions with coordinates restricted to the bounding boxes, with a margin of 16 pixels."""
        if boxes is None:
            self.cropped_shape = volume.shape
            self.crop_slices = tuple(slice(0, s) for s in volume.shape)
            return volume, boxes, point_coords

        crop_margin = 16

        xmin = int(boxes[:, :, 0].min().item())
        ymin = int(boxes[:, :, 1].min().item())
        zmin = int(boxes[:, :, 2].min().item())
        xmax = int(boxes[:, :, 0].max().item())
        ymax = int(boxes[:, :, 1].max().item())
        zmax = int(boxes[:, :, 2].max().item())

        # Make sure margin is respected
        xmin = max(0, xmin - crop_margin)
        ymin = max(0, ymin - crop_margin)
        zmin = max(0, zmin - crop_margin)
        xmax = min(volume.shape[0], xmax + crop_margin)
        ymax = min(volume.shape[1], ymax + crop_margin)
        zmax = min(volume.shape[2], zmax + crop_margin)

        cropped_volume = volume[xmin:xmax, ymin:ymax, zmin:zmax].clone()
        self.cropped_shape = cropped_volume.shape
        self.crop_slices = tuple([slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)])

        cropped_boxes = boxes.clone()
        cropped_boxes[..., 0] = boxes[..., 0] - self.crop_slices[0].start
        cropped_boxes[..., 1] = boxes[..., 1] - self.crop_slices[1].start
        cropped_boxes[..., 2] = boxes[..., 2] - self.crop_slices[2].start

        if point_coords is not None:
            cropped_point_coords = self.crop_point_coords(point_coords)
        else:
            cropped_point_coords = None

        return cropped_volume, cropped_boxes, cropped_point_coords

    def _pad_divisible(self, volume: Image, k: int = 8) -> Image:
        shape_before = volume.shape
        # DivisiblePad expects a 4D tensor, channel first
        padded_volume: Image = self.padder(volume.unsqueeze(0)).squeeze(0)
        shape_after = padded_volume.shape
        self.pad_values = tuple(shape_after[i] - shape_before[i] for i in range(3))
        return padded_volume

    def preprocess(
        self,
        volume: Image,
        boxes: Boxes,
        point_coords: Optional[BatchedPointCoords] = None,
    ) -> tuple[
        Image,
        Boxes,
        Optional[BatchedPointCoords],
    ]:
        self.orig_shape = volume.shape

        # Normalize
        volume = self._normalize_volume(volume)

        # Crop
        volume, boxes, point_coords = self._crop(volume, boxes, point_coords)

        # Downsample
        volume = self.pool_volume(volume)
        boxes = self.pool_boxes(boxes)
        if point_coords is not None:
            point_coords = self.pool_coords(point_coords)

        # Pad to divisible size
        volume = self._pad_divisible(volume)

        return volume, boxes, point_coords

    def forward(
        self,
        volume: Image,
        boxes: Boxes,
        point_coords: Optional[BatchedPointCoords] = None,
    ) -> tuple[
        Image,
        Boxes,
        Optional[BatchedPointCoords],
    ]:
        volume, boxes, point_coords = self.preprocess(volume, boxes, point_coords)

        return volume, boxes, point_coords

    def backward(self, mask_logits: BatchedImageLogits) -> BatchedImageLogits:
        assert self.pad_values is not None, "pad_values must be set before calling backward"
        assert self.orig_shape is not None, "orig_shape must be set before calling backward"
        assert self.crop_slices is not None, "crop_slices must be set before calling backward"
        assert self.cropped_shape is not None, "cropped_shape must be set before calling backward"

        if self.pad_values[0] > 0:
            mask_logits = mask_logits[:, : -self.pad_values[0], :, :]
        if self.pad_values[1] > 0:
            mask_logits = mask_logits[:, :, : -self.pad_values[1], :]
        if self.pad_values[2] > 0:
            mask_logits = mask_logits[:, :, :, : -self.pad_values[2]]

        # F.interpolate expects batch dimension
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=self.cropped_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        pad_sizes = []
        for dim in range(3):
            pad_before = self.crop_slices[dim].start
            pad_after = self.orig_shape[dim] - self.crop_slices[dim].stop
            pad_sizes.extend([pad_after, pad_before])  # NOTE
        pad_sizes = pad_sizes[::-1]  # reverse for torch F.pad

        mask_logits = F.pad(mask_logits, pad_sizes)

        # Upsample back to original size
        # TODO: should not be necessary?
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=self.orig_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return mask_logits
