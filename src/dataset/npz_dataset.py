import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


class NPZDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        size_threshold: int,
        transform: Optional[Callable] = None,
        data_transform: Optional[Callable] = None,
        data_suffix: str = "_resampled.npz",
        load_n_first: Optional[int] = None,
        gt_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            base_dir (str): Root directory to search for image .npz files.
            size_threshold (int): Maximum size of the 3D volume in voxels.
            transform (callable, optional): Optional transform to apply to the samples.
            data_transform (callable, optional): Optional transform to apply to the image data.
            data_suffix (str, optional): File extension to look for (default is '_resampled.npz').
            load_n_first (int, optional): If set, only load the first n files.
            gt_dir (str, optional): Optional directory where corresponding GT .npz files are stored.
            **kwargs: Additional arguments for customization.
        """
        super().__init__()
        self.base_dir: str = base_dir
        self.gt_dir: Optional[str] = gt_dir
        self.size_threshold: int = size_threshold
        self.transform: Optional[Callable] = transform
        self.data_transform: Optional[Callable] = data_transform
        self.data_suffix: str = data_suffix
        self.kwargs: Dict[str, Any] = kwargs

        self.file_paths, self.modalities = self._gather_data()
        if load_n_first is not None:
            self.file_paths = self.file_paths[:load_n_first]
            self.modalities = self.modalities[:load_n_first]

    def _gather_data(self) -> Tuple[List[Tuple[str, Optional[str]]], List[str]]:
        """Collects all file pairs (img_path, gt_path) from base_dir and optional gt_dir."""
        file_paths: List[Tuple[str, Optional[str]]] = []
        modalities: List[str] = []

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(self.data_suffix):
                    continue
                if "limb-Leg" in file or "cremi" in file:
                    continue
                img_path = os.path.join(root, file)

                if self.gt_dir:
                    rel_path = os.path.relpath(img_path, self.base_dir)
                    gt_path = os.path.join(self.gt_dir, rel_path)
                    if not os.path.exists(gt_path):
                        print(f"WARNING: GT file {gt_path} does not exist, skipping this file.")
                        continue
                    file_paths.append((img_path, gt_path))
                else:
                    file_paths.append((img_path, None))

                parts = os.path.normpath(img_path).split(os.sep)
                modality = parts[len(os.path.normpath(self.base_dir).split(os.sep))]
                modalities.append(modality)

        np.random.seed(2025)
        indices = np.random.permutation(len(file_paths))
        file_paths = [file_paths[i] for i in indices]
        modalities = [modalities[i] for i in indices]
        return file_paths, modalities

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, gt_path = self.file_paths[idx]
        img_npz = np.load(img_path, mmap_mode="r")

        if gt_path:
            gt_npz = np.load(gt_path, mmap_mode="r")
            imgs = torch.tensor(img_npz["imgs"])
            gts = torch.tensor(gt_npz["gts"])
            spacing = torch.tensor(gt_npz["spacing"])
        else:
            imgs = torch.tensor(img_npz["imgs"])
            gts = torch.tensor(img_npz["gts"])
            spacing = torch.tensor(img_npz["spacing"])

        unique_labels = np.unique(gts.numpy().astype(np.uint16))
        unique_labels = np.sort(unique_labels)[1:]  # skip background
        if len(unique_labels) == 0:
            print("WARNING: No positive elements in labels file, skipping this file.")
            return self.__getitem__(np.random.randint(len(self)))

        selected_label = np.random.choice(unique_labels)
        labeldata = gts == selected_label
        stacked_data = torch.stack([labeldata, imgs], dim=0)

        z_indices, y_indices, x_indices = torch.where(labeldata)
        if z_indices.numel() == 0:
            print("WARNING: No positive elements in selected label, skipping this file.")
            return self.__getitem__(np.random.randint(len(self)))

        z_min, z_max = z_indices.min().item(), z_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()

        min_offset = 1
        max_offset = 64
        z_min = max(0, z_min - np.random.randint(min_offset, max_offset))
        z_max = min(gts.shape[0] - 1, z_max + np.random.randint(min_offset, max_offset))
        y_min = max(0, y_min - np.random.randint(min_offset, max_offset))
        y_max = min(gts.shape[1] - 1, y_max + np.random.randint(min_offset, max_offset))
        x_min = max(0, x_min - np.random.randint(min_offset, max_offset))
        x_max = min(gts.shape[2] - 1, x_max + np.random.randint(min_offset, max_offset))

        stacked_data = stacked_data[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

        while (stacked_data.shape[1] * stacked_data.shape[2] * stacked_data.shape[3]) > self.size_threshold:
            min_dim_index = int(torch.argmin(torch.tensor(stacked_data.shape[1:])))
            kernel_size = [1, 1, 1]
            kernel_size[min_dim_index] = 2
            stacked_data = (
                F.max_pool3d(stacked_data.unsqueeze(0).float(), kernel_size=kernel_size).squeeze(0).to(torch.uint8)
            )

        if self.transform:
            stacked_data = self.transform(stacked_data)

        labeldata, imgdata = stacked_data.split(1, dim=0)

        imgdata = imgdata.float()
        if imgdata.max() > 0:
            imgdata = imgdata / imgdata.max()
        if self.data_transform:
            imgdata = self.data_transform(imgdata)

        if imgdata.numel() == 0:
            return self.__getitem__(np.random.randint(len(self)))

        # imgdata = imgdata.float()

        rel_path = os.path.relpath(img_path, self.base_dir)
        return {
            "image": imgdata,
            "label": labeldata,
            "boxes": self.get_bboxes_3D(labeldata),
            "spacing": spacing,
            "rel_path": rel_path,
        }

    def get_bbox_2D(self, gt2D: np.ndarray) -> np.ndarray:
        # Compute bounding box for 2D segmentation
        y_indices, x_indices = np.where(gt2D != 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        bbox_shift = np.random.randint(0, 6, 1)[0]
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x / 256)
        bbox_shift_y = int(bbox_shift * scale_y / 256)
        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W - 1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H - 1, y_max + bbox_shift_y)
        boxes = np.array([x_min, y_min, x_max, y_max])
        return boxes

    def get_bboxes_3D(self, gt3D: torch.Tensor) -> torch.Tensor:
        corners_tensor = torch.zeros((2, 3), dtype=torch.float32).to(gt3D.device)
        D, H, W = gt3D.shape[-3:]
        batch_item = gt3D[0]
        z_indices, y_indices, x_indices = np.where(batch_item.cpu() != 0)
        if not len(z_indices):
            print("WARNING: No positive elements in labels file, setting box corners to zero.")
            return corners_tensor
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        z_middle = z_indices[len(z_indices) // 2]
        gt_mid = batch_item[z_middle].cpu()
        box_2d = self.get_bbox_2D(gt_mid)
        x_min, y_min, x_max, y_max = box_2d
        corners_tensor[0, 0] = z_min
        corners_tensor[0, 1] = y_min
        corners_tensor[0, 2] = x_min
        corners_tensor[1, 0] = z_max
        corners_tensor[1, 1] = y_max
        corners_tensor[1, 2] = x_max
        return corners_tensor


def create_weighted_sampler(dataset: NPZDataset) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler so that each modality (first subfolder) is equally likely to be sampled."""
    modality_counts = {}
    for modality in dataset.modalities:
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    num_modalities = len(modality_counts)
    weights = [1.0 / (modality_counts[mod] * num_modalities) for mod in dataset.modalities]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def create_weighted_dataset_folder_sampler(dataset: NPZDataset, epoch_size: int = 1000) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler so that each dataset_folder (second-level folder) is equally likely to be sampled.
    It assumes the file's relative path is of the form 'modality/dataset_folder/file.npz'.
    """
    dataset_folder_counts = {}
    dataset_folders = []
    # For each file, extract the dataset_folder from the relative path.
    for file_path in dataset.file_paths:
        rel_path = os.path.relpath(file_path[0], dataset.base_dir)
        parts = os.path.normpath(rel_path).split(os.sep)
        # If a dataset_folder exists, use it; otherwise fall back to the modality folder.
        dataset_folder = parts[1] if len(parts) >= 2 else parts[0]
        dataset_folders.append(dataset_folder)
        dataset_folder_counts[dataset_folder] = dataset_folder_counts.get(dataset_folder, 0) + 1

    num_dataset_folders = len(dataset_folder_counts)
    weights = [1.0 / (dataset_folder_counts[sub] * num_dataset_folders) for sub in dataset_folders]
    sampler = WeightedRandomSampler(weights, num_samples=epoch_size, replacement=True)
    return sampler
