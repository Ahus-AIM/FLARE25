from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from monai.transforms.croppad.array import DivisiblePad
from tensordict import TensorDict

from src.custom_types import Boxes, Image, MulticlassSegmentation


def dicts_to_tensor_boxes(box_dict: np.ndarray) -> Boxes:
    """
    Convert an array of length n_instances of box dictionaries with the keys "z_min", "z_max", "z_mid", "z_mid_x_min", "z_mid_x_max", "z_mid_y_min", "z_mid_y_max"
    to a tensor of shape (n_instances, 2, 3).
    """
    boxes = torch.zeros((len(box_dict), 2, 3), dtype=torch.int32)
    for i, box in enumerate(box_dict):
        boxes[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]], dtype=torch.int32)
        boxes[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]], dtype=torch.int32)
    return boxes


def crop_to_boxes(
    image: Image,
    multiclass_segmentation: MulticlassSegmentation,
    boxes: Boxes,
    crop_margin: int = 1,
) -> tuple[Image, MulticlassSegmentation, Boxes]:
    """Crop the image and segmentation to the bounding boxes, with a certain margin."""

    if boxes.shape[0] == 0:
        # No boxes, return the original image and segmentation
        return image, multiclass_segmentation, boxes

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
    xmax = min(image.shape[0], xmax + crop_margin)
    ymax = min(image.shape[1], ymax + crop_margin)
    zmax = min(image.shape[2], zmax + crop_margin)

    cropped_image = image[xmin:xmax, ymin:ymax, zmin:zmax].clone()
    cropped_multiclass_segmentation = multiclass_segmentation[xmin:xmax, ymin:ymax, zmin:zmax].clone()

    crop_slices = tuple([slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)])
    cropped_boxes = boxes.clone()
    cropped_boxes[..., 0] = boxes[..., 0] - crop_slices[0].start
    cropped_boxes[..., 1] = boxes[..., 1] - crop_slices[1].start
    cropped_boxes[..., 2] = boxes[..., 2] - crop_slices[2].start

    return cropped_image, cropped_multiclass_segmentation, cropped_boxes


def get_tensordict_iterator(val_dir: Path, val_gt_dir: Path) -> Iterator[TensorDict]:
    """Iterate in a sorted fashion over files in val_dir, yielding TensorDicts of medical data.

    The tensordict has no batch size and the keys
    - "image": the image as a float32 tensor of shape (D, H, W), where each dimension is padded to be divisible by 8
    - "boxes": the boxes as an int32 tensor of shape (n_instances, 2, 3)
    - "spacings": the spacing, as a float32 tensor of shape (3,)
    - "true_multiclass_segmentation": the true multiclass segmentation as a uint8 tensor of shape (D, H, W), where each dimension is padded to be divisible by 8. Read from val_gt_dir.
    """
    file_paths: list[Path] = sorted(val_dir.iterdir())

    padder = DivisiblePad(k=8, method="end", value=0)

    for file_path in file_paths:
        val_data = np.load(file_path, allow_pickle=True)
        val_gt_data = np.load(val_gt_dir / file_path.name, allow_pickle=True)

        # Pad the image and segmentation to be divisible by 8 and scale to [0, 1]
        image = padder(torch.tensor(val_data["imgs"] / 255.0, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        true_multiclass_segmentation = padder(torch.tensor(val_gt_data["gts"], dtype=torch.uint8).unsqueeze(0)).squeeze(
            0
        )

        # Boxes are originally in a dict format
        boxes = (
            dicts_to_tensor_boxes(val_data["boxes"])
            if "boxes" in val_data
            else torch.zeros((0, 2, 3), dtype=torch.int32)
        )

        # Crop the image and segmentation to the bounding boxes
        cropped_image, cropped_true_multiclass_segmentation, cropped_boxes = crop_to_boxes(
            image,
            true_multiclass_segmentation,
            boxes,
        )

        yield TensorDict(
            {
                "image": cropped_image,
                "boxes": (
                    dicts_to_tensor_boxes(val_data["boxes"])
                    if "boxes" in val_data
                    else torch.zeros((0, 2, 3), dtype=torch.int32)
                ),
                "spacing": torch.tensor(val_data["spacing"], dtype=torch.float32),
                "true_multiclass_segmentation": cropped_true_multiclass_segmentation,
            },
            batch_size=(),
        )
