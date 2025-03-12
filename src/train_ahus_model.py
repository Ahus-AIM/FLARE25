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
from monai.losses import DiceCELoss
from torch.backends import cudnn
from tqdm import tqdm

from dataset.npz_dataset import NPZDataset
from model.build_ahus_model import model_registry
from transform.transform import Compose, CropOrPad, Flip
from utils.interact import interact

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
parser.add_argument("--base_dir", type=str, default="/data/drive_data/3D_train_npz_random_10percent_16G")
parser.add_argument("--val_dir", type=str, default="/data/3D_val_npz")
parser.add_argument("--log_every_n_steps", type=int, default=20)
parser.add_argument("--dry_run", action="store_true", default=False)
parser.add_argument("--load_encoder_vit_path", type=str, default="")

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
parser.add_argument("--num_epochs", type=int, default=10_000)
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--accumulation_steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--port", type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    "challenge": interact,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LOGGING_DICT = {}


def save_batch_stats(losses_dict):
    for key, value in losses_dict.items():
        if key not in LOGGING_DICT:
            LOGGING_DICT[key] = []
        LOGGING_DICT[key].append(value)


def ma(arr, k=100):
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
        if not key.startswith("val"):
            continue
        plt.plot(ma(value), label=key, linewidth=0.5, c=f"C{i}")
        plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"{LOG_OUT_DIR}/val_step_loss.png", dpi=300)
    plt.close()
    for i, (key, value) in enumerate(LOGGING_DICT.items()):
        if key.startswith("val"):
            continue
        plt.plot(ma(value), label=key, linewidth=0.5, c=f"C{i}")
        plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"{LOG_OUT_DIR}/train_step_loss.png", dpi=300)
    plt.close()


def save_niigz(volume, save_path):
    if os.path.exists(save_path):
        return
    volume_np = volume.detach().cpu().float().numpy()[0, 0]
    volume_nii = nib.Nifti1Image(volume_np, np.eye(4))
    nib.save(volume_nii, save_path)
    print(f"Saved volume to {save_path}")


def build_model(args):
    return model_registry[args.model_type]().to(device)


def get_dataloaders_npz(args):
    train_dataset = NPZDataset(
        base_dir=args.base_dir,
        transform=Compose([CropOrPad((args.img_size, args.img_size, args.img_size)), Flip()]),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = NPZDataset(
        base_dir=args.val_dir,
        transform=Compose([CropOrPad((args.img_size, args.img_size, args.img_size))]),
        load_n_first=500,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size // 2,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader


class MSEFourierLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xhat, x):
        diff = (xhat - x) ** 2
        fft = torch.fft.fftn(diff, dim=[-3, -2, -1])
        fft = torch.fft.fftshift(fft, dim=[-3, -2, -1])
        fft = torch.abs(fft)
        weight_tensor = torch.zeros_like(fft)

        # Create a high-pass filter
        shape = fft.shape[-3:]  # Get the last three dimensions (spatial dimensions)
        freqs = torch.meshgrid([torch.linspace(-0.5, 0.5, s, device=x.device) for s in shape], indexing="ij")
        freq_radius = torch.sqrt(sum(f**2 for f in freqs))  # Compute frequency magnitude

        weight_tensor = torch.ones_like(fft)
        weight_tensor[freq_radius > 0.1] = 10  # High frequencies get 10x weight

        # Apply the weight
        fft_weighted = (
            fft * weight_tensor
        )  # TODO apply a high pass filter to the fourier transform, so the high frequency components are penalized more (10x)

        loss = torch.mean(fft_weighted)
        return loss


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.train_dataloader, self.val_dataloader = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.val_losses = []
        self.val_dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.resume:
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, "sam_model_latest.pth"))
        else:
            self.init_checkpoint(self.args.checkpoint, self.args.load_encoder_vit_path)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean", smooth_dr=1e-5, smooth_nr=1e-5)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()

    def set_optimizer(self):
        sam_model = self.model

        def get_param_groups(module, lr_scale=1.0, weight_decay=None):
            """Helper function to group parameters while excluding biases and norms from weight decay."""
            decay, no_decay = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if "bias" in name or "norm" in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)

            return [
                {
                    "params": decay,
                    "lr": self.args.lr * lr_scale,
                    "weight_decay": self.args.weight_decay if weight_decay is None else weight_decay,
                },
                {"params": no_decay, "lr": self.args.lr * lr_scale, "weight_decay": 0.0},
            ]

        param_groups = []
        param_groups.extend(get_param_groups(sam_model.segresnet, lr_scale=1.0))
        param_groups.extend(get_param_groups(sam_model.prompt_encoder, lr_scale=1.0))
        param_groups.extend(
            get_param_groups(
                sam_model.mask_decoder,
                lr_scale=1.0,
                weight_decay=0.0 if self.args.model_type.endswith("norm") else self.args.weight_decay,
            )
        )

        self.optimizer = torch.optim.AdamW(
            param_groups, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay
        )

        print("Registering weight normalization post hook")

        def normalize_hook(optimizer, *args, **kwargs):
            for module in sam_model.modules():
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

    def init_checkpoint(self, ckp_path, load_encoder_vit_path=None):
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
                self.dices = last_ckpt["dices"]
                self.best_loss = last_ckpt["best_loss"]
                self.best_dice = last_ckpt["best_dice"]
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

        if load_encoder_vit_path:
            print(f"Loading encoder from {load_encoder_vit_path}")
            vit_mae_ckpt = torch.load(load_encoder_vit_path, map_location=self.args.device, weights_only=False)
            vit_mae_state_dict = vit_mae_ckpt["model_state_dict"]
            encoder_state_dict = {k: v for k, v in vit_mae_state_dict.items() if k.startswith("encoder.")}
            encoder_state_dict = {k.replace("encoder.", ""): v for k, v in encoder_state_dict.items()}
            encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if "neck" not in k}
            self.model.image_encoder.load_state_dict(encoder_state_dict, strict=False)
            # set requires_grad to False for loaded weights (only the keys that are in the loaded weights)
            for name, param in self.model.image_encoder.named_parameters():
                if name in encoder_state_dict:
                    param.requires_grad = False

            print(f"Successfully loaded encoder from {load_encoder_vit_path}")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
            },
            join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"),
        )

    def get_points(self, mask_logits, gt3D, threshold=0.5):
        prediction = (torch.sigmoid(mask_logits) > threshold).long()
        batch_points, batch_labels = click_methods[self.args.click_type](prediction, gt3D)

        if len(batch_points) != mask_logits.shape[0]:
            return None, None

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        return torch.cat(self.click_points, dim=1).to(device), torch.cat(self.click_labels, dim=1).to(device)

    def interaction(self, model, image_embeddings, gt3D, boxes, image3D, xhat):
        losses_dict = {}

        return_loss = self.rec_loss(xhat, image3D)
        losses_dict["rec"] = return_loss.item()

        sparse_emb, dense_emb = model.prompt_encoder(None, boxes, None)
        mask_logits = model.mask_decoder(image_embeddings, model.prompt_encoder.get_dense_pe(), sparse_emb, dense_emb)

        return_loss += self.seg_loss(mask_logits, gt3D)
        losses_dict["box"] = return_loss.item()

        for num_click in range(self.args.num_clicks):
            points_input, labels_input = self.get_points(mask_logits, gt3D, threshold=0.5)
            if points_input is None:
                return_loss += self.seg_loss(mask_logits, gt3D)
                return mask_logits, return_loss, {}

            sparse_emb, dense_emb = model.prompt_encoder((points_input, labels_input), boxes, mask_logits)
            mask_logits = model.mask_decoder(
                image_embeddings, model.prompt_encoder.get_dense_pe(), sparse_emb, dense_emb
            )

            loss = self.seg_loss(mask_logits, gt3D)

            if num_click == args.num_clicks - 1:
                return_loss += args.last_click_loss_weight * loss
            else:
                return_loss += loss

            losses_dict[f"click_{num_click+1}"] = loss.item()

        return mask_logits, return_loss, losses_dict

    def get_dice_score(self, mask_logits, gt3D):
        def compute_dice(mask_pred, mask_gt):
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = mask_logits > 0.0
        true_masks = gt3D > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()
        sam_model = self.model

        tbar = tqdm(self.train_dataloader)

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            if self.args.dry_run and (step > 2):
                break

            try:
                image3D, gt3D, boxes = data3D["image"], data3D["label"], data3D["boxes"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            image3D = image3D.to(device)
            gt3D = (gt3D != 0).to(device).type(torch.long)
            boxes = boxes.to(device)
            with torch.amp.autocast("cuda"):
                image_embeddings, xhat = sam_model.segresnet(image3D)

                self.click_points = []
                self.click_labels = []

                mask_logits, loss, losses_dict = self.interaction(
                    sam_model, image_embeddings, gt3D, boxes, image3D, xhat
                )

            epoch_loss += loss.item()
            epoch_dice += self.get_dice_score(mask_logits, gt3D)
            cur_loss = loss.item()

            loss /= self.args.accumulation_steps

            if torch.isnan(loss):
                print(f"NaN detected in loss at step {step}, skipping batch")
                continue

            self.scaler.scale(loss).backward()
            save_batch_stats(losses_dict)

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(mask_logits, gt3D)
            else:
                step_loss += cur_loss

            if step % self.args.accumulation_steps == 0 and step != 0:
                if print_dice > self.step_best_dice:
                    self.step_best_dice = print_dice
                    if print_dice > 0.9:
                        self.save_checkpoint(
                            epoch,
                            sam_model.state_dict(),
                            describe=f"{epoch}_step_dice:{print_dice}_best",
                        )
                if print_loss < self.step_best_loss:
                    self.step_best_loss = print_loss

            if step % self.args.log_every_n_steps == 0:
                plot_batch_stats()
                os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                save_niigz(mask_logits > 0.0, save_path=f"{LOG_OUT_DIR}/niigz/train_pred.nii.gz")
                save_niigz(torch.sigmoid(mask_logits), save_path=f"{LOG_OUT_DIR}/niigz/train_pred_probs.nii.gz")
                save_niigz(torch.sigmoid(xhat), save_path=f"{LOG_OUT_DIR}/niigz/train_reconstruction.nii.gz")
                save_niigz(gt3D, save_path=f"{LOG_OUT_DIR}/niigz/train_gt.nii.gz")
                save_niigz(image3D, save_path=f"{LOG_OUT_DIR}/niigz/train_image.nii.gz")

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_dice

    def val_epoch(self, epoch):
        epoch_loss = 0
        self.model.eval()
        sam_model = self.model

        tbar = tqdm(self.val_dataloader)

        epoch_dice = 0
        with torch.no_grad():
            for step, data3D in enumerate(tbar):
                if self.args.dry_run and (step > 2):
                    print("Dry run, skipping batch")
                    break

                image3D, gt3D, boxes = data3D["image"], data3D["label"], data3D["boxes"]

                image3D = image3D.to(device)
                gt3D = (gt3D != 0).to(device).type(torch.long)
                boxes = boxes.to(device)
                with torch.amp.autocast("cuda"):
                    image_embeddings, xhat = sam_model.segresnet(image3D)

                    self.click_points = []
                    self.click_labels = []

                    mask_logits, loss, losses_dict = self.interaction(
                        sam_model, image_embeddings, gt3D, boxes, image3D, xhat
                    )

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(mask_logits, gt3D)

                loss /= self.args.accumulation_steps

                save_batch_stats({f"val_{key}": value for key, value in losses_dict.items()})

                if step % self.args.log_every_n_steps == 0:
                    plot_batch_stats()
                    os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                    save_niigz(mask_logits > 0.0, save_path=f"{LOG_OUT_DIR}/niigz/val_pred.nii.gz")
                    save_niigz(torch.sigmoid(mask_logits), save_path=f"{LOG_OUT_DIR}/niigz/val_pred_probs.nii.gz")
                    save_niigz(torch.sigmoid(xhat), save_path=f"{LOG_OUT_DIR}/niigz/val_reconstruction.nii.gz")
                    save_niigz(gt3D, save_path=f"{LOG_OUT_DIR}/niigz/val_gt.nii.gz")
                    save_niigz(image3D, save_path=f"{LOG_OUT_DIR}/niigz/val_image.nii.gz")

            epoch_loss /= step + 1
            epoch_dice /= step + 1

            return epoch_loss, epoch_dice

    def plot_result(self, train_data, val_data, description, save_name):
        plt.plot(train_data)
        plt.plot(val_data)
        plt.legend(["Train", "Val"])
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

            epoch_loss, epoch_dice = self.train_epoch(epoch)
            val_epoch_loss, val_epoch_dice = self.val_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.losses.append(epoch_loss)
            self.dices.append(epoch_dice)
            self.val_losses.append(val_epoch_loss)
            self.val_dices.append(val_epoch_dice)
            print(f"EPOCH: {epoch}, Train Loss: {epoch_loss}, Val Loss: {val_epoch_loss}")
            print(f"EPOCH: {epoch}, Train Dice: {epoch_dice}, Val Dice: {val_epoch_dice}")
            logger.info(f"Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}")

            state_dict = self.model.state_dict()

            # save latest checkpoint
            self.save_checkpoint(epoch, state_dict, describe="latest")
            self.plot_result(self.losses, self.val_losses, "Dice + Cross Entropy Loss", "Loss")
            self.plot_result(self.dices, self.val_dices, "Dice", "Dice")

        logger.info("=====================================================================")
        logger.info(f"Best loss: {self.best_loss}")
        logger.info(f"Best dice: {self.best_dice}")
        logger.info(f"Total loss: {self.losses}")
        logger.info(f"Total dice: {self.dices}")
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
