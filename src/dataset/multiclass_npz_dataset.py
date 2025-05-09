from pathlib import Path

import numpy as np
from tensordict import TensorDict
import torch
from torch.utils.data import Dataset

def dict_to_tensor_boxes(box_dict: dict) -> torch.Tensor:
    """
    Convert a box dictionary with the keys "z_min", "z_max", "z_mid", "z_mid_x_min", "z_mid_x_max", "z_mid_y_min", "z_mid_y_max"
    to a tensor of shape (2, 3).
    """
    boxes = torch.zeros((2, 3), dtype=torch.int32)
    boxes[0, :] = torch.tensor([box_dict["z_min"], box_dict["z_mid_y_min"], box_dict["z_mid_x_min"]])
    boxes[1, :] = torch.tensor([box_dict["z_max"], box_dict["z_mid_y_max"], box_dict["z_mid_x_max"]])
    return boxes


class MulticlassNPZDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.file_paths: list[Path] = list(data_dir.iterdir())

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> TensorDict:
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        return TensorDict({
            "image": torch.tensor(data["image"]),
            "box"
        })
