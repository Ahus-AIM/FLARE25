import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NPZDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        data_suffix: str = "_resampled.npz",
        return_only_image: bool = False,
        load_n_first: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            base_dir (str): Root directory to search for .npz files.
            transform (callable, optional): Optional transform to apply to the samples.
            data_suffix (str, optional): File extension to look for (default is '.npz').
            **kwargs: Additional arguments that can be used for customization.
        """
        super().__init__()
        self.base_dir: str = base_dir
        self.transform: Optional[Callable] = transform
        self.data_suffix: str = data_suffix
        self.file_paths: List[str] = self._gather_data()
        if load_n_first is not None:
            self.file_paths = self.file_paths[:load_n_first]
        self.return_only_image: bool = return_only_image
        self.kwargs: Dict[str, Any] = kwargs

    def _gather_data(self) -> List[str]:
        """Recursively collects all files with the specified suffix in base_dir."""
        file_paths: List[str] = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(self.data_suffix):
                    file_paths.append(os.path.join(root, file))
        np.random.seed(2025)
        np.random.shuffle(file_paths)
        return file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        file_path: str = self.file_paths[idx]
        data: np.lib.npyio.NpzFile = np.load(file_path, mmap_mode="r")

        unique_labels = np.unique(data["gts"].astype(np.uint16))
        unique_labels = np.sort(unique_labels)[1:]
        if not len(unique_labels):
            return self.__getitem__(np.random.randint(len(self)))

        selected_label = np.random.choice(unique_labels)
        labeldata = torch.tensor(data["gts"] == selected_label).long()
        imgdata = torch.tensor(data["imgs"]).float()

        # if labeldata.sum() < 100:
        #     return self.__getitem__(np.random.randint(len(self)))

        rand_seed = np.random.randint(0, 2**32)
        if self.transform:
            labeldata = self.transform(labeldata, rand_seed).unsqueeze(0)
            imgdata = self.transform(imgdata, rand_seed).unsqueeze(0)

        if self.return_only_image:
            return {
                "image": imgdata,
            }

        return {
            "image": imgdata,
            "label": labeldata,
            "boxes": self.get_bboxes_3D(labeldata),
        }

    def get_bbox_2D(self, gt2D):
        # https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/f9ef0731ddbf05b3f1a1399ab4803511168b1e93/get_boxes.py#L44C1-L44C32
        y_indices, x_indices = np.where(gt2D != 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
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

    def get_bboxes_3D(self, gt3D):
        # https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/f9ef0731ddbf05b3f1a1399ab4803511168b1e93/get_boxes.py#L66

        corners_tensor = torch.zeros((2, 3), dtype=torch.float32).to(gt3D.device)
        D, H, W = gt3D.shape[-3:]

        batch_item = gt3D[0]

        z_indices, y_indices, x_indices = np.where(batch_item.cpu() != 0)

        if not len(z_indices):
            print("WARNING: No positive elements in labels file, setting box corners to zero.")
            corners_tensor[0] = 0.0
            corners_tensor[1] = gt3D.shape[-3:] - 1  # NOTE what to do here?

        z_min, z_max = np.min(z_indices), np.max(z_indices)
        z_middle = z_indices[len(z_indices) // 2]

        gt_mid = batch_item[z_middle].cpu()

        box_2d = self.get_bbox_2D(gt_mid)
        x_min, y_min, x_max, y_max = box_2d

        assert z_min == max(0, z_min)
        assert z_max == min(D - 1, z_max)

        corners_tensor[0, 0] = z_min
        corners_tensor[0, 1] = y_min
        corners_tensor[0, 2] = x_min

        corners_tensor[1, 0] = z_max
        corners_tensor[1, 1] = y_max
        corners_tensor[1, 2] = x_max

        return corners_tensor


if __name__ == "__main__":
    # base_dir = "/home/stenheli/drive_data/3D_train_npz_random_10percent_16G"
    base_dir = "../drive_data/3D_train_npz_random_10percent_16G"
    dataset = NPZDataset(base_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(len(dataloader))
    for count, item in enumerate(dataloader):
        for k, v in item.items():
            if k != "spacing":
                print(k, v.shape, v.dtype, len(np.unique(v)))
            else:
                print(k, v)
        print()

        import matplotlib.pyplot as plt

        random_slice = np.random.randint(item["imgs"].shape[1])
        random_int_save_name = np.random.randint(10)

        img = item["imgs"][0, random_slice].float().numpy() / 255
        plt.imsave(f"example_imgs/example_{random_int_save_name}_img.png", img, cmap="gray")
        plt.close()

        lab = item["gts"][0, random_slice].float().numpy() / 255
        plt.imsave(f"example_imgs/example_{random_int_save_name}_gt.png", lab, cmap="nipy_spectral")
        plt.close()

        if count >= 9:
            break
