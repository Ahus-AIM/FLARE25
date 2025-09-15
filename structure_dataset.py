import argparse
import os
import shutil

import tqdm


def move_or_copy_files_in_directory(source_dir: str, target_dir: str, copy: bool) -> None:
    os.makedirs(target_dir, exist_ok=True)

    for file in tqdm.tqdm(os.listdir(source_dir), desc=f"Processing {source_dir}"):
        if not file.endswith(".npz"):
            continue

        parts = file.split("_")
        if len(parts) < 3:
            print("skipping", parts)
            continue  # Skip malformed filenames

        root_dir = f"{parts[0]}_{parts[1]}"
        third_part = parts[2].split("-")[0]
        child_dir = "no_name" if third_part.isdigit() else third_part

        curr_target_dir = os.path.join(target_dir, root_dir, child_dir)
        curr_source_path = os.path.join(source_dir, file)
        curr_target_path = os.path.join(curr_target_dir, file)
        os.makedirs(curr_target_dir, exist_ok=True)

        if copy:
            shutil.copy(curr_source_path, curr_target_path)
        else:
            shutil.move(curr_source_path, curr_target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", default=False, help="Copy files instead of moving them")
    args = parser.parse_args()

    source_target_directories = [
        ("./FLARE-Task1-PancancerRECIST-to-3D/train_npz", "./dataset/FLARE-Task1-PancancerRECIST-to-3D/train"),
        ("./FLARE-Task1-PancancerRECIST-to-3D/val_npz", "./dataset/FLARE-Task1-PancancerRECIST-to-3D/val"),
    ]

    for source_dir, target_dir in source_target_directories:
        move_or_copy_files_in_directory(source_dir, target_dir, copy=args.copy)
