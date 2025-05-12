from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from monai.transforms.croppad.array import DivisiblePad
from tensordict import TensorDict


def dicts_to_tensor_boxes(box_dict: np.ndarray) -> torch.Tensor:
    """
    Convert an array of length n_instances of box dictionaries with the keys "z_min", "z_max", "z_mid", "z_mid_x_min", "z_mid_x_max", "z_mid_y_min", "z_mid_y_max"
    to a tensor of shape (n_instances, 2, 3).
    """
    boxes = torch.zeros((len(box_dict), 2, 3), dtype=torch.int32)
    for i, box in enumerate(box_dict):
        boxes[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]], dtype=torch.int32)
        boxes[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]], dtype=torch.int32)
    return boxes


# TODO: potential improvement: crop to bounding boxes


def get_tensordict_iterator(val_dir: Path, val_gt_dir: Path) -> Iterator[TensorDict]:
    """Iterate in a sorted fashion over files in val_dir, yielding TensorDicts with no batch size, with the keys
    - "image": the image as a float32 tensor of shape (D, H, W), where each dimension is padded to be divisible by 8
    - "boxes": the boxes as an int32 tensor of shape (n_instances, 2, 3)
    - "spacings": the spacing, as a float32 tensor of shape (3,)
    - "true_multiclass_segmentation": the true multiclass segmentation as a uint8 tensor of shape (D, H, W), where each dimension is padded to be divisible by 8. Read from val_gt_dir.

    Note: Since the original ground truth is binary and batched, we assume that no instances are overlapping when we take the argmax.
    """
    file_paths: list[Path] = sorted(val_dir.iterdir())

    padder = DivisiblePad(k=8, method="end", value=0)

    for file_path in file_paths:
        val_data = np.load(file_path, allow_pickle=True)
        val_gt_data = np.load(val_gt_dir / file_path.name, allow_pickle=True)

        yield TensorDict(
            {
                "image": padder(torch.tensor(val_data["imgs"] / 255.0, dtype=torch.float32).unsqueeze(0)).squeeze(0),
                "boxes": (
                    dicts_to_tensor_boxes(val_data["boxes"])
                    if "boxes" in val_data
                    else torch.zeros((0, 2, 3), dtype=torch.int32)
                ),
                "spacing": torch.tensor(val_data["spacing"], dtype=torch.float32),
                "true_multiclass_segmentation": padder(
                    torch.tensor(val_gt_data["gts"], dtype=torch.uint8).unsqueeze(0)
                ).squeeze(0),
            },
            batch_size=(),
        )
