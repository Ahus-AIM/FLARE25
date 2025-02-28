import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torchio as tio
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset


class NPZDataset(Dataset):
    def __init__(
        self, base_dir: str, transform: Optional[Callable] = None, data_suffix: str = ".npz", **kwargs: Any
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
        self.kwargs: Dict[str, Any] = kwargs

    def _gather_data(self) -> List[str]:
        """Recursively collects all files with the specified suffix in base_dir."""
        file_paths: List[str] = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(self.data_suffix):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        file_path: str = self.file_paths[idx]
        data: np.lib.npyio.NpzFile = np.load(file_path)

        unique_labels = np.unique(data["gts"].astype(np.uint16))
        unique_labels = np.sort(unique_labels)[1:]
        selected_label = np.random.choice(unique_labels)
        labeldata = data["gts"] == selected_label

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.tensor(data["imgs"].astype(np.uint8)).unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.tensor(labeldata.astype(np.uint16)).unsqueeze(0)),
        )  # NOTE: spacing is currently not returned

        if self.transform:
            subject = self.transform(subject)

        return {
            "image": subject.image.data.clone().detach(),
            "label": subject.label.data.clone().detach(),
        }


class Union_Dataloader(tio.SubjectsLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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
