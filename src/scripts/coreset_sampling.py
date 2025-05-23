import os
import random
import shutil

from tqdm import tqdm

# Configuration
CONFIG = {
    "source_dir": os.path.normpath("../datasets/CVPR-BiomedSegFM/3D_train_npz_all"),
    "dest_dir": os.path.normpath("../datasets/CVPR_coreset"),
    "exclude_folders": {
        "CT_Aorta",
        "Microscopy_SELMA3D_neural_activity_marker",
        "Microscopy_SELMA3D_nucleus",
    },  # CT_Aorta seems to be normalized in a different way. The two SELMA3D datasets seem to have erroneous labels.
    "total_samples": 4471
    + 983
    + 940,  # This will result in less than 4471 samples as some datasets have only a few samples (meaning it is still an allowed coreset)
}


def is_leaf_folder(path):
    """Return True if `path` contains .npz files and no subfolders."""
    for entry in os.scandir(path):
        if entry.is_dir():
            return False
    return any(f.name.endswith(".npz") for f in os.scandir(path))


def gather_leaf_files(source_dir, exclude):
    """Return dict mapping each leaf folder to its list of .npz file paths."""
    leaf_files = {}
    for root, dirs, files in os.walk(source_dir):
        if any(excl in root for excl in exclude):
            continue
        if is_leaf_folder(root):
            npz_paths = [os.path.join(root, f) for f in files if f.endswith(".npz")]
            if npz_paths:
                leaf_files[root] = npz_paths
    return leaf_files


def prepare_destinations(leaf_folders, source_dir, dest_dir):
    """Create destination folders mirroring structure and return mapping."""
    mapping = {}
    for src in leaf_folders:
        rel = os.path.relpath(src, source_dir)
        dest = os.path.join(dest_dir, rel)
        os.makedirs(dest, exist_ok=True)
        mapping[src] = dest
    return mapping


def sample_and_copy(leaf_files, src_to_dest, per_leaf):
    """Sample up to `per_leaf` files from each leaf and copy them."""
    moved_in_total = 0
    for src_folder, files in leaf_files.items():
        dest_folder = src_to_dest[src_folder]
        # Determine sample count
        count = min(per_leaf, len(files))
        sampled = random.sample(files, count)
        for file_path in tqdm(sampled, desc=f"Copying {os.path.basename(src_folder)}"):
            dest_path = os.path.join(dest_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            moved_in_total += 1
    print(f"Copied {moved_in_total} files in total.")


def main():
    src = CONFIG["source_dir"]
    dst = CONFIG["dest_dir"]
    excl = CONFIG["exclude_folders"]
    total = CONFIG["total_samples"]

    os.makedirs(dst, exist_ok=True)

    # Gather leaf folders
    leaf_files = gather_leaf_files(src, excl)
    num_leaves = len(leaf_files)
    if num_leaves == 0:
        print("No leaf folders found.")
        return

    # Compute per-leaf sample size
    per_leaf = total // num_leaves
    print(f"Sampling up to {per_leaf} files from each of {num_leaves} leaf folders.")

    # Prepare destination mapping
    src_to_dest = prepare_destinations(leaf_files.keys(), src, dst)

    # Sample and copy files
    sample_and_copy(leaf_files, src_to_dest, per_leaf)

    print("Coreset sampling complete.")


if __name__ == "__main__":
    main()
