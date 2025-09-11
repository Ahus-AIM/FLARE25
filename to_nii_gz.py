import os

import nibabel as nib
import numpy as np


def convert_npz_to_nii_gz(input_folder, output_folder=None):
    """
    Convert all .npz files in a folder to .nii.gz format.

    Args:
        input_folder (str): Path to folder containing .npz files.
        output_folder (str): Path to save converted .nii.gz files. Defaults to input folder.
    """
    if output_folder is None:
        output_folder = input_folder

    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".npz"):
            npz_path = os.path.join(input_folder, file_name)
            data = np.load(npz_path)

            array_key = "gts"
            image_data = data[array_key]

            nii_image = nib.Nifti1Image(image_data, affine=np.eye(4))  # Identity affine by default

            out_name = os.path.splitext(file_name)[0] + ".nii.gz"
            out_path = os.path.join(output_folder, out_name)
            nib.save(nii_image, out_path)
            print(f"Converted: {file_name} -> {out_name}")


if __name__ == "__main__":
    input_folder = "./inputs"
    output_folder = "./inputs_nii_gz"  # Or None to save in same folder
    convert_npz_to_nii_gz(input_folder, output_folder)
