from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tensordict import TensorDict


def dicts_to_tensor_boxes(box_dict: np.ndarray) -> torch.Tensor:
    """
    Convert an array of length n_instances of box dictionaries with the keys "z_min", "z_max", "z_mid", "z_mid_x_min", "z_mid_x_max", "z_mid_y_min", "z_mid_y_max"
    to a tensor of shape (n_instances, 2, 3).
    """
    boxes = torch.zeros((len(box_dict), 2, 3), dtype=torch.int32)
    for i, box in enumerate(box_dict):
        boxes[i, 0, :] = torch.tensor(
            [box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]], dtype=torch.int32
        )
        boxes[i, 1, :] = torch.tensor(
            [box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]], dtype=torch.int32
        )
    return boxes


def get_tensordict_iterator(data_dir: Path) -> Iterator[TensorDict]:
    """Return an iterator over the files in the data_dir, yielding TensorDicts with no batch size, with the keys
    - "image": the image as a float32 tensor of shape (D, H, W)
    - "boxes": the boxes as an int32 tensor of shape (n_instances, 2, 3)
    - "spacings": the spacing for each instance, as a float32 tensor of shape (n_instances, 3)
    - "true_multiclass_segmentation": the true multiclass segmentation as a uint8 tensor of shape (D, H, W)
    """
    file_paths: list[Path] = list(data_dir.iterdir())

    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)
        yield TensorDict(
            {
                "image": torch.tensor(data["imgs"] / 255.0, dtype=torch.float32),
                "boxes": dicts_to_tensor_boxes(data["boxes"]),
                "spacings": torch.tensor(data["spacing"], dtype=torch.float32),
                "true_multiclass_segmentation": torch.tensor(
                    data["gts"], dtype=torch.uint8
                ),
            },
            batch_size=(),
        )
