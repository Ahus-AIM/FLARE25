import argparse
import cProfile
import os

import torch

from src.train_ahus_model import main


def _main():

    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise RuntimeError(
            "CUDA_LAUNCH_BLOCKING is not set. Please run 'export CUDA_LAUNCH_BLOCKING=1' before running this script."
        )

    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="union_train")
    parser.add_argument("--click_type", type=str, default="challenge")
    parser.add_argument("--model_type", type=str, default="ahus_model_sinusoidal")
    parser.add_argument("--checkpoint", type=str, default="ckpt/sam_med3d.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--num_clicks", type=int, default=2)
    parser.add_argument("--last_click_loss_weight", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default="/data/drive_data/3D_train_npz_random_10percent_16G_original")
    parser.add_argument("--val_dir", type=str, default="/data/3D_val_npz")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=True)
    parser.add_argument("--size_threshold", type=int, default=128 * 128 * 128)

    # train
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)

    # lr_scheduler
    parser.add_argument("--lr_scheduler", type=str, default="multisteplr")
    parser.add_argument("--step_size", type=list, default=[120, 180])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=12361)

    args = parser.parse_args()

    trainer = main(args, train=False)
    trainer.scaler = torch.amp.GradScaler("cuda")

    cProfile.run("trainer.train_epoch(0)", sort="tottime")


if __name__ == "__main__":
    _main()
