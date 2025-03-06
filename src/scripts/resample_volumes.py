import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


def normalize_volume(volume):
    volume[volume <= 0] = torch.nan
    positive_volume = volume[~torch.isnan(volume)]
    min_val = positive_volume.min()
    max_val = positive_volume.max()
    volume = (volume - min_val + 1) / (max_val - min_val + 1)
    volume[torch.isnan(volume)] = 0
    return volume


def resample_volume(npz, full_path, image_size=128, save_nii_gz=False):
    """
    Resample the image and label volumes from the npz file to have 1mm isotropic spacing,
    and save the resampled image (and label) as NIfTI (.nii.gz) files.

    Parameters:
      npz (dict): Dictionary loaded from an .npz file. Expected keys:
          - "imgs": 3D image volume (assumed shape: (z, y, x))
          - "gts": 3D label volume (assumed shape: (z, y, x))
          - "spacing": Voxel spacing given as (x, y, z)
      full_path (str): Full path (including filename) where the resampled image should be saved.
    """
    # Load image and label volumes
    image = torch.tensor(npz["imgs"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    label = torch.tensor(npz["gts"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    original_spacing = torch.tensor(npz["spacing"], dtype=torch.float32)
    if any(original_spacing <= 0):
        print(f"Skipping {full_path} due to invalid spacing: {original_spacing}")
        return

    # Normalize spacing to image_size
    dim1 = original_spacing[2] * image.shape[2]
    dim2 = original_spacing[1] * image.shape[3]
    dim3 = original_spacing[0] * image.shape[4]
    max_dim = max(dim1, dim2, dim3)
    original_spacing = original_spacing / max_dim * image_size

    effective_spacing = torch.tensor([original_spacing[2], original_spacing[1], original_spacing[0]])
    target_spacing = torch.tensor([1, 1, 1])
    new_shape = torch.round(torch.tensor(image.shape[2:]) * effective_spacing / target_spacing).int()

    # Do not downsample the "thin" dimension if not necessary
    for i in range(3):
        new_shape[i] = new_shape[i].clamp(min=image.shape[2 + i], max=torch.tensor(image_size))

    resampled_image = (
        F.interpolate(image, size=new_shape.tolist(), mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
    )
    resampled_image = normalize_volume(resampled_image)
    resampled_label = F.interpolate(label, size=new_shape.tolist(), mode="nearest").squeeze(0).squeeze(0)

    if save_nii_gz:
        affine = np.eye(4)
        img_path = full_path.replace(".npz", "_img.nii.gz")
        nii_img = nib.Nifti1Image(resampled_image.numpy(), affine)
        label_path = full_path.replace(".nii.gz", "_label.nii.gz")
        nii_lbl = nib.Nifti1Image(resampled_label.numpy(), affine)
        nib.save(nii_img, img_path)
        nib.save(nii_lbl, label_path)
    else:
        np.savez(
            full_path,
            imgs=resampled_image.numpy(),
            gts=resampled_label.numpy(),
            spacing=target_spacing.numpy(),
        )


if __name__ == "__main__":
    walk_dir = "../drive_data/3D_train_npz_random_10percent_16G"
    for root, dirs, files in tqdm.tqdm(os.walk(walk_dir)):
        num_instances = []
        shapes = []
        for file in files:
            if file.endswith(".npz") and not file.endswith("_resampled.npz"):
                full_path = os.path.join(root, file)
                npz = np.load(full_path)

                resample_volume(
                    npz,
                    os.path.join(root, file.replace(".npz", "_resampled.npz")),
                    image_size=128,
                )
