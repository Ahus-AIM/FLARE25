import argparse
import os
from typing import List

import numpy as np


def read_input_files(folder: str) -> List[str]:
    return [file for file in os.listdir(folder)]


def predict(x):
    if "prev_pred" in x:
        return x["prev_pred"]
    pred = np.zeros(x["imgs"].shape)
    for i in range(len(x["boxes"])):
        x_min, x_max = x["boxes"][i]["z_mid_x_min"], x["boxes"][i]["z_mid_x_max"] + 1
        y_min, y_max = x["boxes"][i]["z_mid_y_min"], x["boxes"][i]["z_mid_y_max"] + 1
        z_min, z_max = x["boxes"][i]["z_min"], x["boxes"][i]["z_max"] + 1
        pred[z_min:z_max, y_min:y_max, x_min:x_max] = np.full((z_max - z_min, y_max - y_min, x_max - x_min), i + 1)
    return pred


def load_predict_and_save(load_path: str, save_path: str):
    files = read_input_files(load_path)
    for file in files:
        curr_file_path = os.path.join(load_path, file)
        x = np.load(curr_file_path, allow_pickle=True)
        prediction = predict(x)

        save_file_path = os.path.join(save_path, file)
        np.savez(save_file_path, segs=prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the segmentation of the input images.")
    parser.add_argument("--load_path", default="./workspace/inputs", type=str, help="Path to the input images.")
    parser.add_argument("--save_path", default="./workspace/outputs", type=str, help="Path to save the predictions.")

    args = parser.parse_args()

    load_path = args.load_path
    save_path = args.save_path

    load_predict_and_save(load_path, save_path)
