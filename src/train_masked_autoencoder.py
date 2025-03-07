# set up environment
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np

join = os.path.join
import argparse

import nibabel as nib
import torch
import torch.multiprocessing as mp
from torch.backends import cudnn
from tqdm import tqdm

from dataset.npz_dataset import NPZDataset
from model.build_sam3D import sam_model_registry3D
from transform.transform import Compose, CropOrPad, Flip

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="union_train")
parser.add_argument("--click_type", type=str, default="random")
parser.add_argument("--model_type", type=str, default="vit_b_ori")
parser.add_argument("--checkpoint", type=str, default="ckpt/sam_med3d.pth")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--work_dir", type=str, default="work_dir")
parser.add_argument("--num_clicks", type=int, default=5)
parser.add_argument("--last_click_loss_weight", type=int, default=1)
parser.add_argument("--base_dir", type=str, default="../drive_data/3D_train_npz_random_10percent_16G")
parser.add_argument("--val_dir", type=str, default="/data/3D_val_npz")
parser.add_argument("--log_every_n_steps", type=int, default=20)
parser.add_argument("--dry_run", action="store_true", default=False)

# train
parser.add_argument("--num_workers", type=int, default=24)
parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
parser.add_argument("--multi_gpu", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--allow_partial_weight", action="store_true", default=False)

# lr_scheduler
parser.add_argument("--lr_scheduler", type=str, default="multisteplr")
parser.add_argument("--step_size", type=list, default=[120, 180])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=2000)
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--accumulation_steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--port", type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LOGGING_DICT = {}


def save_batch_stats(losses_dict):
    for key, value in losses_dict.items():
        if key not in LOGGING_DICT:
            LOGGING_DICT[key] = []
        LOGGING_DICT[key].append(value)


def ma(arr, k=50):
    res = []
    curr = np.mean(arr[:k])
    for i in range(len(arr)):
        if i < k:
            res.append(np.mean(arr[:i]))
        else:
            curr = 1 / k * arr[i] + (1 - 1 / k) * curr
            res.append(curr)
    return res


def plot_batch_stats():
    for i, (key, value) in enumerate(LOGGING_DICT.items()):
        plt.plot(
            torch.linspace(0, 1, len(value)), ma(value), label=f"{key} {len(value)} steps", linewidth=0.5, c=f"C{i}"
        )
        plt.grid(True)
        plt.yscale("log")
    plt.legend()
    plt.savefig(f"{LOG_OUT_DIR}/step_loss.png", dpi=300)
    plt.close()


def save_niigz(volume, save_path):
    if os.path.exists(save_path):
        return
    volume_np = volume.detach().cpu().float().numpy()[0, 0]
    volume_nii = nib.Nifti1Image(volume_np, np.eye(4))
    nib.save(volume_nii, save_path)
    print(f"Saved volume to {save_path}")


def build_model(args):
    vit_mae = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    return vit_mae


def get_dataloaders_npz(args):
    train_dataset = NPZDataset(
        base_dir=args.base_dir,
        transform=Compose([CropOrPad((args.img_size, args.img_size, args.img_size)), Flip()]),
        return_only_image=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_dataset = NPZDataset(
        base_dir=args.val_dir,
        transform=Compose([CropOrPad((args.img_size, args.img_size, args.img_size)), Flip()]),
        return_only_image=True,
        load_n_first=500,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_dataloader, val_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.train_dataloader, self.val_dataloader = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.step_best_loss = np.inf
        self.losses = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.resume:
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, "vit_mae_latest.pth"))
        else:
            self.init_checkpoint(self.args.checkpoint)

    def set_loss_fn(self):
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def set_optimizer(self):
        vit_mae = self.model

        self.optimizer = torch.optim.AdamW(
            vit_mae.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay
        )

        if self.args.model_type.endswith("norm"):
            print("Registering weight normalization post hook")

            def normalize_hook(optimizer, *args, **kwargs):
                for module in vit_mae.modules():
                    if hasattr(module, "normalize_weights"):
                        module.normalize_weights()

            self.optimizer.register_step_post_hook(normalize_hook)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.args.step_size, self.args.gamma
            )
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.step_size[0], self.args.gamma)
        elif self.args.lr_scheduler == "coswarm":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if self.args.allow_partial_weight:
                self.model.load_state_dict(last_ckpt["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(last_ckpt["model_state_dict"])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt["epoch"]
                self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(last_ckpt["lr_scheduler_state_dict"])
                self.losses = last_ckpt["losses"]
                self.best_loss = last_ckpt["best_loss"]
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "best_loss": self.best_loss,
                "args": self.args,
            },
            join(MODEL_SAVE_PATH, f"vit_mae_{describe}.pth"),
        )

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()
        vit_mae = self.model

        tbar = tqdm(self.train_dataloader)

        self.optimizer.zero_grad()
        step_loss = 0
        for step, data3D in enumerate(tbar):
            if self.args.dry_run and (step > 2):
                break
            try:
                x = data3D["image"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")

            x = x.to(device)
            with torch.amp.autocast("cuda"):
                x_hat = vit_mae(x)
                loss = self.mse_loss(x_hat, x)

            epoch_loss += loss.item()
            cur_loss = loss.item()

            loss /= self.args.accumulation_steps

            if torch.isnan(loss):
                print(f"NaN detected in loss at step {step}, skipping batch")
                continue

            self.scaler.scale(loss).backward()
            save_batch_stats({"train_loss": loss.item()})

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
            else:
                step_loss += cur_loss

            if step % self.args.accumulation_steps == 0 and step != 0:
                if print_loss < self.step_best_loss:
                    self.step_best_loss = print_loss

            if step % self.args.log_every_n_steps == 0:
                plot_batch_stats()
                os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                save_niigz(x, save_path=f"{LOG_OUT_DIR}/niigz/image.nii.gz")
                save_niigz(x_hat, save_path=f"{LOG_OUT_DIR}/niigz/reconstruction.nii.gz")
                save_niigz(x, save_path=f"{LOG_OUT_DIR}/niigz/image.nii.gz")
                save_niigz(torch.abs(x - x_hat), save_path=f"{LOG_OUT_DIR}/niigz/error.nii.gz")

        epoch_loss /= step + 1

        return epoch_loss

    def val_epoch(self, epoch):
        epoch_loss = 0
        self.model.eval()
        vit_mae = self.model

        tbar = tqdm(self.val_dataloader)

        with torch.no_grad():
            for step, data3D in enumerate(tbar):
                if self.args.dry_run and (step > 2):
                    break
                try:
                    x = data3D["image"]
                except Exception as e:
                    print(f"Error processing batch at step {step}: {e}")

                x = x.to(device)
                with torch.amp.autocast("cuda"):
                    x_hat = vit_mae(x)
                    loss = self.mse_loss(x_hat, x)

                epoch_loss += loss.item()

                loss /= self.args.accumulation_steps

                save_batch_stats({"val_loss": loss.item()})

                if step % self.args.log_every_n_steps == 0:
                    plot_batch_stats()
                    os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                    save_niigz(x, save_path=f"{LOG_OUT_DIR}/niigz/val_image.nii.gz")
                    save_niigz(x_hat, save_path=f"{LOG_OUT_DIR}/niigz/val_reconstruction.nii.gz")
                    save_niigz(torch.abs(x - x_hat), save_path=f"{LOG_OUT_DIR}/niigz/val_error.nii.gz")

            epoch_loss /= step + 1

            return epoch_loss

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel("Epoch")
        plt.ylabel(f"{save_name}")
        plt.savefig(join(MODEL_SAVE_PATH, f"{save_name}.png"))
        plt.close()

    def train(self):
        if self.args.dry_run:
            self.args.num_epochs = 1
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")

            train_epoch_loss = self.train_epoch(epoch)
            val_epoch_loss = self.val_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.losses.append(train_epoch_loss)
            print(f"EPOCH: {epoch}, Train Loss: {train_epoch_loss}, Val Loss: {val_epoch_loss}")
            logger.info(f"Epoch\t {epoch}\t : train loss: {train_epoch_loss}, val loss: {val_epoch_loss}")

            state_dict = self.model.state_dict()

            # save latest checkpoint
            self.save_checkpoint(epoch, state_dict, describe="latest")

            # save train loss best checkpoint
            if train_epoch_loss < self.best_loss:
                self.best_loss = train_epoch_loss
                self.save_checkpoint(epoch, state_dict, describe="loss_best")

            self.plot_result(self.losses, "MSE Loss", "Loss")
        logger.info("=====================================================================")
        logger.info(f"Best loss: {self.best_loss}")
        logger.info(f"Total loss: {self.losses}")
        logger.info("=====================================================================")
        logger.info(f"args : {self.args}")
        logger.info("=====================================================================")


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if args.device == "mps":
            args.device = torch.device("mps")
        else:
            args.device = torch.device(f"cuda:{args.gpu_ids[0]}")

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy("file_system")
    device_config(args)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    # Load datasets
    dataloaders = get_dataloaders_npz(args)
    # Build model
    model = build_model(args)
    # Create trainer
    trainer = BaseTrainer(model, dataloaders, args)
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
