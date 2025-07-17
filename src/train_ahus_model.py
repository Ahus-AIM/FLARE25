# set up environment
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np

join = os.path.join
import argparse
from pathlib import Path
from typing import Callable, Optional, overload

import nibabel as nib
import torch
import torch.multiprocessing as mp
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    CropForeground,
    OneOf,
    RandBiasField,
    RandFlip,
    RandGaussianSmooth,
    RandHistogramShift,
    Transform,
)
from torch.backends import cudnn
from torch.optim import AdamW, Optimizer
from tqdm import tqdm

from src.dataset.npz_dataset import NPZDataset, create_weighted_dataset_folder_sampler, create_weighted_sampler
from src.model.registry import model_registry
from src.utils.decode import decoder_forward
from src.utils.interact import random_interact as interact

LOGGING_DICT = {}
CLASS_STATS_DICT = {"train": {}, "val": {}}
LOG_OUT_DIR = "log_dir"
MODEL_SAVE_PATH = "model_save_path"
click_methods = {
    "challenge": interact,
}
sampler_class = {
    "modality": create_weighted_sampler,
    "dataset": create_weighted_dataset_folder_sampler,
}

logger = logging.getLogger(__name__)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self, params: list[torch.Tensor], lr: float, momentum: float, weight_decay: float):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.weight_decay = weight_decay

    @overload
    def step(self, closure: None = ...) -> None: ...  # noqa: E704

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...  # noqa: E704

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update

                if self.weight_decay != 0:
                    p.data.mul_(1 - lr * self.weight_decay)  # apply weight decay

                p.data.add_(update, alpha=-lr)  # take a step
        return loss


class AdamMuon(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        muon_params, adam_params = [], []
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            for group in params:
                for p in group["params"]:
                    if p.ndim > 1:
                        muon_params.append(p)
                    else:
                        adam_params.append(p)
        else:
            for p in params:
                if p.ndim > 1:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        self.muon = Muon(muon_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay) if muon_params else None
        self.adam = AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) if adam_params else None

        # Combine param groups for compatibility with PyTorch schedulers
        param_groups = []
        if self.muon:
            for g in self.muon.param_groups:
                g["optimizer"] = "muon"
                param_groups.append(g)
        if self.adam:
            for g in self.adam.param_groups:
                g["optimizer"] = "adam"
                param_groups.append(g)

        defaults = dict(lr=lr, momentum=muon_momentum, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...  # noqa: E704

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...  # noqa: E704

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam:
            self.adam.zero_grad(set_to_none=set_to_none)


def save_batch_stats(losses_dict):
    for key, value in losses_dict.items():
        if key not in LOGGING_DICT:
            LOGGING_DICT[key] = []
        LOGGING_DICT[key].append(value)


def save_class_stats(dataset_type, class_stats_dict, idx):
    if dataset_type not in CLASS_STATS_DICT:
        CLASS_STATS_DICT[dataset_type] = {}

    for loss_type, class_type in class_stats_dict.items():
        if loss_type not in CLASS_STATS_DICT[dataset_type]:
            CLASS_STATS_DICT[dataset_type][loss_type] = {}

        for class_name, value in class_type.items():
            if class_name not in CLASS_STATS_DICT[dataset_type][loss_type]:
                CLASS_STATS_DICT[dataset_type][loss_type][class_name] = [[], []]

            CLASS_STATS_DICT[dataset_type][loss_type][class_name][0].append(value)
            CLASS_STATS_DICT[dataset_type][loss_type][class_name][1].append(idx)


def ma(arr, k=300):
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


def _plot_class_stats(group, prefix):
    linestyles = ["-", "--", "-.", ":"] * 1000
    num_colors = 10
    for loss_type, class_type in group.items():
        for i, (key, (value, value_idx)) in enumerate(sorted(class_type.items())):
            plt.plot(
                value_idx,
                ma(value),
                label=key,
                linewidth=0.5,
                c=f"C{i % num_colors}",
                linestyle=linestyles[i // num_colors],
            )
            plt.grid(True)
        plt.legend(bbox_to_anchor=(1.00, 1.0), loc="upper left")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{LOG_OUT_DIR}/{prefix}_{loss_type}_class_stats.png", dpi=300)
        plt.close()


def plot_class_stats():
    for dataset_type, dataset_type_values in CLASS_STATS_DICT.items():
        groups = {}
        for k, v in dataset_type_values.items():
            groups[k] = {}
            for name, (value, idx) in v.items():
                group_name = "-".join(name[:-1])
                instance_name = name[-1]
                if group_name not in groups[k]:
                    groups[k][group_name] = {}
                groups[k][group_name][instance_name] = [value, idx]
        for group_name, group in groups.items():
            _plot_class_stats(group, f"{dataset_type}_{group_name}")


def save_latest_niigz_files(nii_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for class_path, class_dict in nii_dict.items():
        class_path_name = "/".join(class_path)
        os.makedirs(f"{out_dir}/{class_path_name}", exist_ok=True)
        save_niigz(
            class_dict["mask_logits"] > 0.0,
            save_path=f"{out_dir}/{class_path_name}/logits_pred.nii.gz",
            overwrite=True,
        )
        save_niigz(
            torch.sigmoid(class_dict["mask_logits"]),
            save_path=f"{out_dir}/{class_path_name}/probs_pred.nii.gz",
            overwrite=True,
        )
        save_niigz(
            class_dict["mask_targets"],
            save_path=f"{out_dir}/{class_path_name}/targets.nii.gz",
            overwrite=True,
        )
        save_niigz(
            class_dict["image"],
            save_path=f"{out_dir}/{class_path_name}/image.nii.gz",
            overwrite=True,
        )


def save_niigz(volume, save_path, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        return
    volume_np = volume.detach().cpu().float().numpy()
    volume_np = volume_np[*[0] * (len(volume_np.shape) - 3)]
    volume_nii = nib.Nifti1Image(volume_np, np.eye(4))
    nib.save(volume_nii, save_path)
    print(f"Saved volume to {save_path}")


class RandPermuteAxes(Transform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1).item() < self.prob:
            spatial_dims = list(range(1, img.ndim))  # Exclude batch/channel dim
            permuted_dims = torch.randperm(len(spatial_dims)).tolist()  # Get a random permutation
            img = img.permute(0, *(spatial_dims[i] for i in permuted_dims))  # Reorder spatial axes
        return img


class RandInvertColors(Transform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1).item() < self.prob:
            img = 1.0 - img  # Invert colors
            img = img.clamp(0.0, 1.0)  # Ensure values are in [0, 1]
        return img


class ClampTransform(Transform):
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        return torch.clamp(img, self.min_val, self.max_val)


def build_model(args):
    return model_registry[args.model_type]().to(args.device)


def get_dataloaders_npz(args):
    transform_prob = 0.5
    rand_transforms = Compose(
        [
            OneOf(
                [
                    OneOf(
                        [
                            RandBiasField(prob=transform_prob),
                            RandGaussianSmooth(prob=transform_prob),
                            RandHistogramShift(prob=transform_prob),
                        ]
                    ),
                    RandInvertColors(prob=0.0),
                ]
            ),
            ClampTransform(min_val=0.0, max_val=1.0),
        ]
    )

    threshold_value = 0
    transform = Compose(
        [
            CropForeground(select_fn=lambda x: x > threshold_value, k_divisible=16, allow_smaller=True),
            RandFlip(spatial_axis=0, prob=0.5),  # Random flip along axis 0
            RandFlip(spatial_axis=1, prob=0.5),  # Random flip along axis 1
            RandFlip(spatial_axis=2, prob=0.5),  # Random flip along axis 2
            # RandPermuteAxes(prob=1.0),
        ]
    )
    val_transform = CropForeground(select_fn=lambda x: x > 0, k_divisible=16, allow_smaller=True)

    print(args.train_dir)
    train_dataset = NPZDataset(
        base_dir=args.train_dir,
        transform=transform,
        size_threshold=args.size_threshold,
        data_suffix="npz",
        # data_transform=rand_transforms,
    )
    train_sampler = sampler_class[args.data_sampling_method](train_dataset, epoch_size=10000)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        # shuffle=True,
    )

    val_dataset = NPZDataset(
        base_dir=args.val_img_dir,
        transform=val_transform,
        size_threshold=args.size_threshold,
        data_suffix="npz",
        gt_dir=None,
        validation=True,
    )
    # val_sampler = sampler_class[args.data_sampling_method](val_dataset, epoch_size=None)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # sampler=val_sampler,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


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
        self.modality_stats = {}
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.resume:
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, "model_latest.pth"))
        else:
            self.init_checkpoint(self.args.checkpoint)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean", smooth_dr=1e-5, smooth_nr=1e-5, lambda_ce=2.0
        )

    def set_optimizer(self):
        model = self.model

        params = model.parameters()
        self.optimizer = AdamMuon(params, weight_decay=self.args.weight_decay, lr=self.args.lr)

        print("Registering weight normalization post hook")

        def normalize_hook(optimizer, *args, **kwargs):
            for module in model.modules():
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
                self.dices = last_ckpt["dices"]
                self.best_loss = last_ckpt["best_loss"]
                self.best_dice = last_ckpt["best_dice"]
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
                "dices": self.dices,
                "val_losses": self.val_losses,
                "val_dices": self.val_dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "stats_dict": LOGGING_DICT,
                "class_stats_dict": CLASS_STATS_DICT,
            },
            join(MODEL_SAVE_PATH, f"model_{describe}.pth"),
        )

    def get_points(self, mask_logits, mask_targets, threshold=0.5):
        prediction = (torch.sigmoid(mask_logits) > threshold).long()
        device = prediction.device
        batch_points, batch_labels = click_methods[self.args.click_type](prediction, mask_targets)

        if len(batch_points) != mask_logits.shape[0]:
            return None, None

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        return torch.cat(self.click_points, dim=1).to(device), torch.cat(self.click_labels, dim=1).to(device)

    def store_class_losses_and_nii(self, image, mask_logits, mask_targets, rel_file_path, split_filename_to_dirs):
        if split_filename_to_dirs:
            root_paths = []
            sub_paths = []
            for p in rel_file_path:
                root_paths.append(tuple(p.split("_")[:1]))
                sub_paths.append(tuple(p.split("_")[:2]))
        else:
            root_paths = [Path(p).parts[:1] for p in rel_file_path]
            sub_paths = [Path(p).parts[:2] for p in rel_file_path]

        root_paths_set = set(root_paths)
        sub_paths_set = set(sub_paths)

        nii_dict = {}

        seg_losses_dict = {}

        tot_running_seg_loss = 0

        for root_path in root_paths_set:
            curr_root_seg_loss = 0
            num_samples = 0

            for sub_path in sub_paths_set:
                if not sub_path[0] == root_path[0]:
                    continue
                idxs = [i for i, p in enumerate(sub_paths) if p == sub_path]
                num_samples += len(idxs)

                curr_seg_loss = self.seg_loss(mask_logits[idxs], mask_targets[idxs])
                curr_root_seg_loss += curr_seg_loss * len(idxs)
                seg_losses_dict[sub_path] = curr_seg_loss.detach().cpu().numpy()

                nii_dict[sub_path] = {
                    "mask_logits": mask_logits[idxs[-1]].detach().cpu(),
                    "mask_targets": mask_targets[idxs[-1]].detach().cpu(),
                    "image": image[idxs[-1]].detach().cpu(),
                }

            tot_running_seg_loss += curr_root_seg_loss
            seg_losses_dict[root_path] = curr_root_seg_loss.detach().cpu().numpy() / num_samples

        loss = tot_running_seg_loss / len(sub_paths)
        class_losses_dict = {"seg": seg_losses_dict}

        return loss, class_losses_dict, nii_dict

    def interaction(
        self,
        model,
        image_embeddings,
        mask_targets,
        boxes,
        image,
        rel_file_path=None,
        split_filename_to_dirs=False,
        zero_pos_weight=1e-1,
    ):
        losses_dict = {}

        pos_weight = torch.where(
            torch.isclose(image, torch.zeros(image.shape, device=image.device)), zero_pos_weight, 1
        )
        reweighing = pos_weight.numel() / pos_weight.sum()
        pos_weight_reweighing = pos_weight * reweighing

        mask_logits, _ = decoder_forward(model, image_embeddings, mask_logits=None, points=None, boxes=boxes)

        if boxes is None:
            loss, class_losses_dict, nii_dict = self.store_class_losses_and_nii(
                image, mask_logits, mask_targets * 0, rel_file_path, split_filename_to_dirs
            )
            losses_dict["no_box"] = loss.item()
        else:
            loss, class_losses_dict, nii_dict = self.store_class_losses_and_nii(
                image, mask_logits, mask_targets, rel_file_path, split_filename_to_dirs
            )
            losses_dict["box"] = loss.item()

        return_loss = loss

        for num_click in range(self.args.num_clicks):
            points_input, labels_input = self.get_points(mask_logits, mask_targets, threshold=0.5)
            if points_input is None:
                return_loss += self.seg_loss(mask_logits, mask_targets)
                return mask_logits, loss, {}, class_losses_dict, nii_dict
            mask_logits, _ = decoder_forward(model, image_embeddings, mask_logits, (points_input, labels_input), boxes)
            loss = self.seg_loss(mask_logits, mask_targets)

            return_loss += loss

            losses_dict[f"click_{num_click+1}"] = loss.item()

        return mask_logits, return_loss, losses_dict, class_losses_dict, nii_dict

    def get_dice_score(self, mask_logits, mask_targets):
        def compute_dice(mask_pred, mask_gt):
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = mask_logits > 0.0
        true_masks = mask_targets > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        dice = sum(dice_list) / len(dice_list)
        if hasattr(dice, "item"):
            return dice.item()
        return dice

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()
        model = self.model
        model = model.train()
        device = next(model.parameters()).device

        tbar = tqdm(self.train_dataloader)
        nii_dict = {}

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            if self.args.dry_run and (step > 2):
                break

            if self.args.profile and (step > 10):
                print("Profiling finished.")
                return

            try:
                image, mask_targets, boxes, rel_file_path = (
                    data3D["image"],
                    data3D["label"],
                    data3D["boxes"],
                    data3D["rel_path"],
                )
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            try:
                image = image.to(device)
                mask_targets = (mask_targets != 0).to(device).type(torch.long)
                boxes = boxes.to(device)
                # if boxes are all zeros, set to None
                if boxes.abs().sum() == 0:
                    boxes = None
                with torch.amp.autocast("cuda"):
                    image_embeddings = model.segresnet(image)

                    self.click_points = []
                    self.click_labels = []

                    mask_logits, loss, losses_dict, class_losses_dict, curr_nii_dict = self.interaction(
                        model, image_embeddings, mask_targets, boxes, image, rel_file_path
                    )
                nii_dict = nii_dict | curr_nii_dict

                # limit nii dict to 10 items
                if len(nii_dict) > 10:
                    nii_dict = {k: nii_dict[k] for k in list(nii_dict)[-10:]}

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(mask_logits, mask_targets)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                if torch.isnan(loss):
                    print(f"NaN detected in loss at step {step}, skipping batch")
                    continue

                self.scaler.scale(loss).backward()
                save_batch_stats(losses_dict)
                save_class_stats("train", class_losses_dict, step + epoch * len(self.train_dataloader))

                if step % self.args.accumulation_steps == 0 and step != 0:
                    # clip grads at magnitude 1
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    print_loss = step_loss / self.args.accumulation_steps
                    step_loss = 0
                    print_dice = self.get_dice_score(mask_logits, mask_targets)
                else:
                    step_loss += cur_loss

                if step % self.args.accumulation_steps == 0 and step != 0:
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(
                                epoch,
                                model.state_dict(),
                                describe=f"{epoch}_step_dice:{print_dice}_best",
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

                if step % self.args.log_every_n_steps == 0 and not self.args.profile:
                    plot_batch_stats()
                    plot_class_stats()
                    os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                if step % (self.args.log_every_n_steps * 5) == 0 and not self.args.profile:
                    save_niigz(mask_logits > 0.0, save_path=f"{LOG_OUT_DIR}/niigz/train_pred.nii.gz")
                    save_niigz(torch.sigmoid(mask_logits), save_path=f"{LOG_OUT_DIR}/niigz/train_pred_probs.nii.gz")
                    save_niigz(mask_targets, save_path=f"{LOG_OUT_DIR}/niigz/train_gt.nii.gz")
                    save_niigz(image, save_path=f"{LOG_OUT_DIR}/niigz/train_image.nii.gz")
                    save_latest_niigz_files(nii_dict, f"{LOG_OUT_DIR}/niigz/train")
            except Exception as e:
                print(f"Error during training at step {step}: {e}")
                continue

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_dice

    def val_epoch(self, epoch):
        epoch_loss = 0
        self.model.eval()
        model = self.model
        model = model.eval()
        device = next(model.parameters()).device

        tbar = tqdm(self.val_dataloader)

        nii_dict = {}

        epoch_dice = 0
        with torch.no_grad():
            for step, data3D in enumerate(tbar):
                if self.args.dry_run and (step > 2):
                    print("Dry run, skipping batch")
                    break

                image, mask_targets, boxes, rel_file_path = (
                    data3D["image"],
                    data3D["label"],
                    data3D["boxes"],
                    data3D["rel_path"],
                )

                image = image.to(device)
                mask_targets = (mask_targets != 0).to(device).type(torch.long)
                boxes = boxes.to(device)
                if boxes.abs().sum() == 0:
                    boxes = None
                with torch.amp.autocast("cuda"):
                    image_embeddings = model.segresnet(image)

                    self.click_points = []
                    self.click_labels = []

                    mask_logits, loss, losses_dict, class_losses_dict, curr_nii_dict = self.interaction(
                        model,
                        image_embeddings,
                        mask_targets,
                        boxes,
                        image,
                        rel_file_path,
                        split_filename_to_dirs=True,
                    )

                nii_dict = nii_dict | curr_nii_dict

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(mask_logits, mask_targets)

                loss /= self.args.accumulation_steps

                save_batch_stats({f"val_{key}": value for key, value in losses_dict.items()})
                save_class_stats("val", class_losses_dict, step + epoch * len(self.val_dataloader))

                if step % self.args.log_every_n_steps == 0:
                    plot_batch_stats()
                    plot_class_stats()
                    os.makedirs(f"{LOG_OUT_DIR}/niigz", exist_ok=True)
                if step % (self.args.log_every_n_steps * 20) == 0:
                    save_niigz(mask_logits > 0.0, save_path=f"{LOG_OUT_DIR}/niigz/val_pred.nii.gz")
                    save_niigz(torch.sigmoid(mask_logits), save_path=f"{LOG_OUT_DIR}/niigz/val_pred_probs.nii.gz")
                    save_niigz(mask_targets, save_path=f"{LOG_OUT_DIR}/niigz/val_gt.nii.gz")
                    save_niigz(image, save_path=f"{LOG_OUT_DIR}/niigz/val_image.nii.gz")
                    save_latest_niigz_files(nii_dict, f"{LOG_OUT_DIR}/niigz/val")

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

            # try:
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
            # except Exception as e:
            #     print(f"Error during training/validation at epoch {epoch}: {e}")
            #     continue

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
    torch.manual_s1eed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        args.device = torch.device(args.device)
    except ValueError as e:
        raise ValueError(f"Invalid device argument: {e}")


def main(args, train=True):
    os.makedirs(LOG_OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    mp.set_sharing_strategy("file_system")
    device_config(args)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    # Load datasets
    dataloaders = get_dataloaders_npz(args)
    # Build model
    model = build_model(args)
    print(f"Model numels: {sum(p.numel() for p in model.parameters())}")
    # Create trainer
    trainer = BaseTrainer(model, dataloaders, args)
    # Train
    if train:
        trainer.train()
    return trainer


if __name__ == "__main__":
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="union_train")
    parser.add_argument("--click_type", type=str, default="challenge")
    parser.add_argument("--model_type", type=str, default="ahus_model_rope_mixed")
    parser.add_argument("--checkpoint", type=str, default="ckpt/sam_med3d.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--num_clicks", type=int, default=0)
    parser.add_argument("--last_click_loss_weight", type=int, default=1)
    parser.add_argument("--train_dir", type=str, default="/dataset/FLARE-MedFM/train/")
    parser.add_argument("--val_img_dir", type=str, default="/dataset/FLARE-MedFM/val/")
    # parser.add_argument(
    #    "--val_gt_dir", type=str, default="../datasets/CVPR-BiomedSegFM/3D_val_gt/3D_val_gt_interactive"
    # )
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--size_threshold", type=int, default=128**3)
    parser.add_argument("--data_sampling_method", type=str, default="dataset")

    # train
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)

    # lr_scheduler
    parser.add_argument("--lr_scheduler", type=str, default="multisteplr")
    parser.add_argument(
        "--step_size",
        type=list,
        default=[
            5,
            10,
            15,
            20,
        ],
    )  # 20, 40, 80])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--port", type=int, default=12361)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
    LOG_OUT_DIR = join(args.work_dir, args.task_name)
    MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
    main(args)
