import argparse
import random
import shutil
from pathlib import Path


def copy_random_files_with_rest(src_folder, dst_folder, rest_folder, num_files=10):
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)
    rest_path = Path(rest_folder)

    # Ensure source exists and is a directory
    if not src_path.is_dir():
        raise NotADirectoryError(f"{src_path} is not a valid directory")

    # Get all files (non-recursively)
    all_files = [f for f in src_path.iterdir() if f.is_file()]

    if len(all_files) < num_files:
        raise ValueError(f"Source folder only contains {len(all_files)} files, but {num_files} requested")

    # Pick random files for the subset
    selected_files = random.sample(all_files, num_files)

    # Determine the rest of the files
    rest_files = set(all_files) - set(selected_files)

    # Create destination folders if they don't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    rest_path.mkdir(parents=True, exist_ok=True)

    # Copy selected files to the subset folder
    for file in selected_files:
        shutil.copy(file, dst_path / file.name)

    # Copy the rest of the files to the rest folder
    for file in rest_files:
        shutil.copy(file, rest_path / file.name)

    print(f"Copied {num_files} files to {dst_path}")
    print(f"Copied {len(rest_files)} files to {rest_path}")


def main(args):
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    rest_folder = args.rest_folder
    num_files = args.num_files

    copy_random_files_with_rest(src_folder, dst_folder, rest_folder, num_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy a random subset of files from one directory to another, and save the rest to a separate folder."
    )
    parser.add_argument("src_folder", type=str, help="Source folder containing files.")
    parser.add_argument("dst_folder", type=str, help="Destination folder to copy the subset of files to.")
    parser.add_argument("rest_folder", type=str, help="Destination folder to copy the rest of the files to.")
    parser.add_argument(
        "--num_files", type=int, default=10, help="Number of random files to copy to the subset folder."
    )

    args = parser.parse_args()
    main(args)
