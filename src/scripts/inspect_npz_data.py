import os
import random

import nibabel as nib
import numpy as np

# Define source and destination paths
SOURCE_DIR = "/data/drive_data/3D_train_npz_random_10percent_16G_original"
DEST_DIR = "./tmp_folders/"

os.makedirs(DEST_DIR, exist_ok=True)


# Function to copy the directory structure
def copy_structure(src, dest):
    for root, dirs, _ in os.walk(src):
        rel_path = os.path.relpath(root, src)
        new_dir = os.path.join(dest, rel_path)
        os.makedirs(new_dir, exist_ok=True)


# Function to find all .npz files in a directory
def find_npz_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npz")]


# Function to convert and save a random npz sample as .nii.gz
def save_random_nii(src_folder, dest_folder):
    print(f"Processing {src_folder}")
    npz_files = find_npz_files(src_folder)
    if not npz_files:
        return  # No files to process

    # Select a random file
    random_file = random.choice(npz_files)
    npz_path = os.path.join(src_folder, random_file)

    # Load NPZ file
    data = np.load(npz_path)
    if "imgs" not in data:
        print(f'Error: "img" key not found in {npz_path}')
        return

    img_array = data["imgs"]
    label_array = data["gts"]
    img_shape = img_array.shape
    img_shape = f"{img_shape[0]}x{img_shape[1]}x{img_shape[2]}"

    unique_labels = np.unique(label_array)

    # Convert to NIfTI format
    nii_filename = os.path.join(dest_folder, f"random_sample_{img_shape}.nii.gz")
    nii_img = nib.Nifti1Image(img_array, affine=np.eye(4))
    nib.save(nii_img, nii_filename)
    nii_filename = os.path.join(dest_folder, f"random_label_{len(unique_labels)}.nii.gz")
    nii_img = nib.Nifti1Image(label_array, affine=np.eye(4))
    nib.save(nii_img, nii_filename)


# Copy the folder structure
copy_structure(SOURCE_DIR, DEST_DIR)

# Iterate over subdirectories and process .npz files
for root, dirs, _ in os.walk(SOURCE_DIR):
    for dir_name in dirs:
        src_folder = os.path.join(root, dir_name)
        dest_folder = os.path.join(DEST_DIR, os.path.relpath(src_folder, SOURCE_DIR))
        save_random_nii(src_folder, dest_folder)
