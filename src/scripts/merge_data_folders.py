import argparse
import os

import numpy as np
from tqdm import tqdm


def merge_npz_files(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Wrap the file listing with tqdm for progress
    for filename in tqdm(os.listdir(folder1), desc="Processing files"):
        if not filename.endswith(".npz"):
            continue

        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        if not os.path.exists(path2):
            tqdm.write(f"Warning: {filename} not found in {folder2}")
            continue

        data1 = np.load(path1, allow_pickle=True)
        data2 = np.load(path2, allow_pickle=True)

        merged = {}

        # Copy everything from data1
        for key in data1.files:
            merged[key] = data1[key]

        # Copy everything from data2, unless already present
        for key in data2.files:
            if key not in merged:
                merged[key] = data2[key]
            elif key in ("boxes", "spacing"):
                if not np.array_equal(data1[key], data2[key]):
                    tqdm.write(f"⚠️ Mismatch in key '{key}' for file '{filename}'")

        out_path = os.path.join(output_folder, filename)
        np.savez_compressed(out_path, **merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge matching .npz files from two folders into a third one.")
    parser.add_argument("folder1", help="Path to first folder (e.g., with imgs/text_prompts)")
    parser.add_argument("folder2", help="Path to second folder (e.g., with gts)")
    parser.add_argument("output_folder", help="Path to output folder for merged .npz files")
    args = parser.parse_args()

    merge_npz_files(args.folder1, args.folder2, args.output_folder)
