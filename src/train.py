# set up environment
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np

join = os.path.join
import argparse
from contextlib import nullcontext

import nibabel as nib
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.backends import cudnn
from tqdm import tqdm

from dataset.npz_dataset import NPZDataset
from model.build_sam3D import sam_model_registry3D
from utils.click_method import get_clicks_for_class_error, get_next_click3D_torch_2

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="union_train")
parser.add_argument("--click_type", type=str, default="random")
parser.add_argument("--multi_click", action="store_true", default=False)
parser.add_argument("--model_type", type=str, default="vit_b_ori")
parser.add_argument("--checkpoint", type=str, default="ckpt/sam_med3d.pth")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--work_dir", type=str, default="work_dir")
parser.add_argument("--num_clicks", type=int, default=5)
parser.add_argument("--last_click_loss_weight", type=int, default=1)
parser.add_argument("--base_dir", type=str, default="../drive_data/3D_train_npz_random_10percent_16G")
parser.add_argument("--log_every_n_steps", type=int, default=20)

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
parser.add_argument("--num_epochs", type=int, default=200)
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
click_methods = {
    "random": get_next_click3D_torch_2,
    "challenge": get_clicks_for_class_error,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LOGGING_DICT = {}


def save_batch_stats(losses_dict):
    for key, value in losses_dict.items():
        if key not in LOGGING_DICT:
            LOGGING_DICT[key] = []
        LOGGING_DICT[key].append(value)


def ma(arr, k=500):
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
        plt.plot(ma(value), label=key, linewidth=0.5, c=f"C{i}")
        plt.grid(True)
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
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    return sam_model


# def get_dataloaders(args):
#     train_dataset = Dataset_Union_ALL(
#         paths=img_datas,
#         transform=tio.Compose(
#             [
#                 # tio.ToCanonical(),
#                 # tio.CropOrPad(
#                 #     # mask_name="label",
#                 #     target_shape=(args.img_size, args.img_size, args.img_size),
#                 # ),  # crop only object region
#                 tio.RandomFlip(axes=(0, 1, 2)),
#             ]
#         ),
#         threshold=1000,
#     )

#     if args.multi_gpu:
#         train_sampler = DistributedSampler(train_dataset)
#         shuffle = False
#     else:
#         train_sampler = None
#         shuffle = True

#     train_dataloader = Union_Dataloader(
#         dataset=train_dataset,
#         sampler=train_sampler,
#         batch_size=args.batch_size,
#         shuffle=shuffle,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     return train_dataloader


def get_dataloaders_npz(args):
    train_dataset = NPZDataset(
        base_dir=args.base_dir,
        transform=tio.Compose(
            [
                # tio.ToCanonical(),
                tio.CropOrPad(
                    # mask_name="label",
                    target_shape=(args.img_size, args.img_size, args.img_size),
                ),
                tio.RandomFlip(axes=(0, 1, 2)),
            ]
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.resume:
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, "sam_model_latest.pth"))
        else:
            self.init_checkpoint(self.args.checkpoint)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

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
        param_groups.extend(get_param_groups(sam_model.image_encoder, lr_scale=1.0))
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

        if self.args.model_type.endswith("norm"):
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
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                # "used_datas": img_datas,
            },
            join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"),
        )

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None, boxes=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        if len(batch_points) != prev_masks.shape[0]:
            return None, None

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        return points_co, points_la

    def get_bbox_2D(self, gt2D):
        # https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/f9ef0731ddbf05b3f1a1399ab4803511168b1e93/get_boxes.py#L44C1-L44C32
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        bbox_shift = np.random.randint(0, 6, 1)[0]
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x / 256)
        bbox_shift_y = int(bbox_shift * scale_y / 256)

        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W - 1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H - 1, y_max + bbox_shift_y)
        boxes = np.array([x_min, y_min, x_max, y_max])
        return boxes

    def get_bboxes_3D(self, gt3D):
        # https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/f9ef0731ddbf05b3f1a1399ab4803511168b1e93/get_boxes.py#L66

        corners_tensor = torch.zeros((gt3D.shape[0], 2, 3), dtype=torch.float32).to(gt3D.device)
        D, H, W = gt3D.shape[-3:]

        for i in range(gt3D.shape[0]):
            batch_item = gt3D[i, 0]

            z_indices, y_indices, x_indices = np.where(batch_item.cpu() > 0)

            if not len(z_indices):
                print("WARNING: No positive elements in labels file, setting box corners to zero.")
                corners_tensor[i, 0] = 0.0
                corners_tensor[i, 1] = 100.0  # NOTE what to do here?
                continue

            z_min, z_max = np.min(z_indices), np.max(z_indices)
            z_middle = z_indices[len(z_indices) // 2]

            gt_mid = batch_item[z_middle].cpu()

            box_2d = self.get_bbox_2D(gt_mid)
            x_min, y_min, x_max, y_max = box_2d

            assert z_min == max(0, z_min)
            assert z_max == min(D - 1, z_max)

            corners_tensor[i, 0, 0] = z_min
            corners_tensor[i, 0, 1] = y_min
            corners_tensor[i, 0, 2] = x_min

            corners_tensor[i, 1, 0] = z_max
            corners_tensor[i, 1, 1] = y_max
            corners_tensor[i, 1, 2] = x_max

        return corners_tensor

    def interaction(self, sam_model, image_embedding, gt3D):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
        )
        random_insert = np.random.randint(2, 9)
        for num_click in range(self.args.num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == self.args.num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None
                )
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D,
                    low_res_masks,
                    points=[points_input, labels_input],
                )
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def interaction_modified(self, sam_model, image_embedding, gt3D):
        losses_dict = {}

        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
        )

        initial_boxes = self.get_bboxes_3D(gt3D)

        low_res_masks, prev_masks = self.batch_forward(
            sam_model,
            image_embedding,
            gt3D,
            low_res_masks,
            points=None,
            boxes=initial_boxes,
        )

        return_loss = self.seg_loss(prev_masks, gt3D)
        losses_dict["box"] = return_loss.item()

        for num_click in range(self.args.num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)
            if points_input is None:
                return_loss = self.seg_loss(prev_masks, gt3D)
                return prev_masks, return_loss, {}

            low_res_masks, prev_masks = self.batch_forward(
                sam_model,
                image_embedding,
                gt3D,
                low_res_masks,
                points=[points_input, labels_input],
                boxes=initial_boxes,
            )
            loss = self.seg_loss(prev_masks, gt3D)
            if num_click == args.num_clicks - 1:
                return_loss += args.last_click_loss_weight * loss
            else:
                return_loss += loss

            losses_dict[f"click_{num_click+1}"] = loss.item()

        return prev_masks, return_loss, losses_dict

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = prev_masks > 0.0
        true_masks = gt3D > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, args):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        sam_model = self.model
        self.args.rank = -1

        tbar = tqdm(self.dataloaders)

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D["image"], data3D["label"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            my_context = (
                self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext
            )

            with my_context():
                image3D = image3D.to(device)
                gt3D = (gt3D != 0).to(device).type(torch.long)
                with torch.amp.autocast("cuda"):
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss, losses_dict = self.interaction_modified(sam_model, image_embedding, gt3D)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()
                save_batch_stats(losses_dict)

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if step % self.args.accumulation_steps == 0 and step != 0:
                # print(f"Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}")
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
                save_niigz(prev_masks > 0.0, save_path=f"{LOG_OUT_DIR}/niigz/pred.nii.gz")
                save_niigz(torch.sigmoid(prev_masks), save_path=f"{LOG_OUT_DIR}/niigz/predSigmoid.nii.gz")
                save_niigz(gt3D, save_path=f"{LOG_OUT_DIR}/niigz/gt.nii.gz")
                save_niigz(image3D, save_path=f"{LOG_OUT_DIR}/niigz/image.nii.gz")

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_iou, epoch_dice, pred_list

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
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")

            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, self.args.num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.losses.append(epoch_loss)
            self.dices.append(epoch_dice)
            print(f"EPOCH: {epoch}, Loss: {epoch_loss}")
            print(f"EPOCH: {epoch}, Dice: {epoch_dice}")
            logger.info(f"Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}")

            state_dict = self.model.state_dict()

            # save latest checkpoint
            self.save_checkpoint(epoch, state_dict, describe="latest")

            # save train loss best checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(epoch, state_dict, describe="loss_best")

            # save train dice best checkpoint
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                self.save_checkpoint(epoch, state_dict, describe="dice_best")

            self.plot_result(self.losses, "Dice + Cross Entropy Loss", "Loss")
            self.plot_result(self.dices, "Dice", "Dice")
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
