import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_tag_data(names, file_paths):
    dfs = []
    for name, path in zip(names, file_paths):
        df = pd.read_csv(path)
        df["Model"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_metrics(data):
    metrics = ["TotalRunningTime", "DSC_AUC", "NSD_AUC", "DSC_Final", "NSD_Final"]
    sns.set(style="whitegrid", context="talk")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        sns.boxplot(x="Model", y=metric, data=data, ax=axes[idx], palette="Set2")
        axes[idx].set_title(metric)

    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def main(args):
    data = load_and_tag_data(args.names, args.csv_paths)
    plot_metrics(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation model performance.")
    parser.add_argument("csv_paths", nargs="+", help="List of CSV file paths for each model.")
    parser.add_argument("--names", nargs="+", help="List of names for each model.")
    args = parser.parse_args()
    main(args)
