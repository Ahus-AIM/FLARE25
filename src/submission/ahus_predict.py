"""
This command is expected to take (D, H, W) images from one folder and write (D, H, W) segmentations (integer valued) to a different folder
"""

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Integer
from monai.transforms import DivisiblePad
from tensordict import TensorDict

from src.model.build_ahus_model import model_registry
from src.rl.utils import pad_prompt_embeddings
from src.submission.segmentation import Segmenter, segmenter_registry
from src.utils.decode import decoder_forward

torch.set_grad_enabled(False)


class VolumeTransforms:
    def __init__(self, size_threshold: int) -> None:
        self.size_threshold = size_threshold
        self.pooling_factors = [1, 1, 1]
        self.crop_slices = None
        self.pad_values = None
        self.orig_shape = None
        self.cropped_shape = None
        self.padder = DivisiblePad(k=8, method="end", value=0)

    @staticmethod
    def _normalize_volume(volume: torch.Tensor) -> torch.Tensor:
        volume = volume.clone().float()
        volume[volume <= 0] = torch.nan
        positive_volume = volume[~torch.isnan(volume)]
        if positive_volume.numel() == 0:
            return torch.zeros_like(volume)
        min_val = positive_volume.min()
        max_val = positive_volume.max()
        volume = (volume - min_val + 1) / (max_val - min_val + 1)
        volume[torch.isnan(volume)] = 0
        return volume

    def _adaptive_max_pool(self, volume: torch.Tensor, boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        shape = list(volume.shape[2:])  # D, H, W
        self.pooling_factors = [1, 1, 1]
        while volume.numel() > self.size_threshold:
            min_dim = int(np.argmin(shape))
            kernel_size = [2, 2, 2]
            kernel_size[min_dim] = 1
            volume = F.max_pool3d(volume, kernel_size=kernel_size)
            shape = list(volume.shape[2:])
            self.pooling_factors = [self.pooling_factors[i] * kernel_size[i] for i in range(3)]

        boxes = boxes / torch.tensor(self.pooling_factors, device=boxes.device).view(1, 1, 3)
        if points is not None:
            points[0] = points[0] / torch.tensor(self.pooling_factors, device=points[0].device).view(1, 1, 3)
        return volume, boxes, points

    def _crop_volume(self, volume: torch.Tensor, boxes: torch.tensor, points: torch.Tensor) -> torch.Tensor:
        crop_margin = 16

        xmin = int(boxes[:, :, 0].min().item())
        ymin = int(boxes[:, :, 1].min().item())
        zmin = int(boxes[:, :, 2].min().item())
        xmax = int(boxes[:, :, 0].max().item())
        ymax = int(boxes[:, :, 1].max().item())
        zmax = int(boxes[:, :, 2].max().item())
        xmin = max(0, xmin - crop_margin)
        ymin = max(0, ymin - crop_margin)
        zmin = max(0, zmin - crop_margin)
        xmax = min(volume.shape[2], xmax + crop_margin)
        ymax = min(volume.shape[3], ymax + crop_margin)
        zmax = min(volume.shape[4], zmax + crop_margin)

        cropped = volume[:, :, xmin:xmax, ymin:ymax, zmin:zmax]
        self.crop_slices = [slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)]

        boxes[..., 0] = boxes[..., 0] - self.crop_slices[0].start
        boxes[..., 1] = boxes[..., 1] - self.crop_slices[1].start
        boxes[..., 2] = boxes[..., 2] - self.crop_slices[2].start
        if points is not None:
            points[0][..., 0] = points[0][..., 0] - self.crop_slices[0].start
            points[0][..., 1] = points[0][..., 1] - self.crop_slices[1].start
            points[0][..., 2] = points[0][..., 2] - self.crop_slices[2].start

        self.cropped_shape = cropped.shape[2:]

        return cropped, boxes, points

    def _pad_divisible(self, volume: torch.Tensor, k: int = 8) -> torch.Tensor:
        shape_before = volume.shape[2:]
        volume = self.padder(volume.squeeze(0)).unsqueeze(0)
        shape_after = volume.shape[2:]
        self.pad_values = [shape_after[i] - shape_before[i] for i in range(3)]
        return volume

    def preprocess_volume(self, volume: torch.Tensor, boxes: torch.Tensor, points=torch.Tensor) -> torch.Tensor:
        volume = volume.clone()
        # clone boxes since _crop_volume modifies them in-place
        boxes = boxes.clone()

        self.orig_shape = volume.shape[-3:]

        # Normalize
        volume = self._normalize_volume(volume)

        # Crop and track slices.
        volume, boxes, points = self._crop_volume(volume, boxes, points)

        # Downsample with adaptive max pooling
        volume, boxes, points = self._adaptive_max_pool(volume, boxes, points)

        # Pad to divisible size
        volume = self._pad_divisible(volume)

        return volume, boxes, points

    # def transform_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns coordinates in the cropped/pooled space.
    #     """
    #     coords = coords.clone()
    #     for i in range(3):
    #         if self.crop_slices is not None:
    #             coords[..., i] -= self.crop_slices[i].start# / self.pooling_factors[i]
    #         coords[..., i] = coords[..., i] / self.pooling_factors[i]
    #     return coords

    def forward(
        self,
        volume: torch.Tensor,
        spacing: np.ndarray = None,
        points: Optional[List[torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
    ):
        volume, boxes, points = self.preprocess_volume(volume, boxes, points)

        return volume, points, boxes

    def backward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        if self.pad_values[0] > 0:
            mask_logits = mask_logits[:, :, : -self.pad_values[0], :, :]
        if self.pad_values[1] > 0:
            mask_logits = mask_logits[:, :, :, : -self.pad_values[1], :]
        if self.pad_values[2] > 0:
            mask_logits = mask_logits[:, :, :, :, : -self.pad_values[2]]

        mask_logits = F.interpolate(mask_logits, size=self.cropped_shape, mode="trilinear", align_corners=False)

        pad_sizes = []
        for dim in range(3):
            pad_before = self.crop_slices[dim].start
            pad_after = self.orig_shape[dim] - self.crop_slices[dim].stop
            pad_sizes.extend([pad_after, pad_before])  # NOTE

        pad_sizes = pad_sizes[::-1]  # reverse for torch F.pad
        mask_logits = F.pad(mask_logits, pad_sizes)

        # Upsample back to original size
        return F.interpolate(mask_logits, size=self.orig_shape, mode="trilinear", align_corners=False)


def safe_autocast(device_type: str):
    if device_type == "cuda":
        return torch.autocast(device_type=device_type)
    elif device_type == "cpu":
        return torch.autocast(device_type=device_type)
    else:
        return nullcontext()


class InferencePipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        # Model device is prioritized over segmenter device for general operations
        self.model_device: str = args.model_device
        self.segmenter_device: str = args.segmenter_device
        self.n_steps: int = args.n_steps
        self.model: torch.nn.Module = self._load_model()
        self.segmenter: Segmenter = self._load_segmenter()
        self.coord_handler: VolumeTransforms = VolumeTransforms(args.size_threshold)

    # -------------------- Data Loading Helpers -------------------- #
    def _get_auxiliary_path(self, main_file: str, prefix: str) -> str:
        """
        Build the auxiliary file path by adding a prefix to the main file's basename.
        """
        parts: List[str] = main_file.split(os.sep)
        aux_path: str = os.sep.join(parts[:-1]) + os.sep + f"{prefix}{parts[-1]}"
        return aux_path

    def _load_npz(self, file_path: str) -> Dict[str, Any]:
        """
        Load an NPZ file and return its contents as a dictionary.
        """
        with np.load(file_path, allow_pickle=True) as data:
            return dict(data)

    def _save_npz(self, file_path: str, **data: Any) -> None:
        """
        Save data to an NPZ file.
        """
        np.savez(file_path, **data)

    def _handle_boxes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add 'boxes' information if present.
          - If already in data, save it to an auxiliary file.
          - If in a auxiliary file, load it.
          - Otherwise, create a bbox covering the whole image.
        """
        boxes_path: str = self._get_auxiliary_path(self.full_file, "boxes_")
        if "boxes" in data:
            self._save_npz(boxes_path, boxes=data["boxes"])
        elif os.path.exists(boxes_path):
            boxes_data: Dict[str, Any] = self._load_npz(boxes_path)
            data["boxes"] = boxes_data["boxes"]
        else:  # create a bbox covering the whole image (1 pixel margin)
            image_shape = data["imgs"].shape
            data["boxes"] = [
                {
                    # First point
                    "z_min": 0,
                    "z_mid_y_min": 0,
                    "z_mid_x_min": 0,
                    # Second point
                    "z_max": image_shape[0] - 1,
                    "z_mid_y_max": image_shape[1] - 1,
                    "z_mid_x_max": image_shape[2] - 1,
                }
            ]
            self._save_npz(boxes_path, boxes=data["boxes"])

        return data

    def _handle_mask_logits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        If a file with 'mask_logits_' prefix exists, use it to override mask_logits in data.
        """
        mask_logits_path: str = self._get_auxiliary_path(self.full_file, "mask_logits_")
        if os.path.exists(mask_logits_path):
            logits_data: Dict[str, Any] = self._load_npz(mask_logits_path)
            data["mask_logits"] = torch.tensor(logits_data["mask_logits"])
        return data

    def load_data(self) -> Dict[str, Any]:
        """
        Load the main data file and ensure that boxes and mask logits are available.
        """
        data: Dict[str, Any] = self._load_npz(self.full_file)
        data = self._handle_mask_logits(data)
        data = self._handle_boxes(data)
        return data

    # -------------------- Inference Helpers -------------------- #
    def _transform(self, image3D_np: np.ndarray) -> torch.Tensor:
        volume: torch.Tensor = torch.tensor(image3D_np).float().unsqueeze(0).unsqueeze(0)
        return volume

    def _get_initial_boxes(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Converts the boxes from the data dictionary into a tensor format.
        """
        boxes: Any = data["boxes"]
        boxes_tensor: torch.Tensor = torch.zeros((len(boxes), 2, 3), dtype=torch.float32)
        for i, box in enumerate(boxes):
            boxes_tensor[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]])
            boxes_tensor[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]])
        return boxes_tensor.to(self.model_device)

    def _get_points(self, data: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
        """
        Converts the points from the data dictionary into a tensor format.
        """
        if "clicks" not in data:
            return None

        num_clicks: int = len(data["clicks"][0]["fg"]) + len(data["clicks"][0]["bg"])
        points: torch.Tensor = torch.zeros((len(data["clicks"]), num_clicks, 3), dtype=torch.float32)
        click_types: torch.Tensor = torch.zeros((len(data["clicks"]), num_clicks), dtype=torch.long)

        for i, click in enumerate(data["clicks"]):
            all_clicks: List[Any] = click["fg"] + click["bg"]
            for j, point in enumerate(all_clicks):
                points[i, j, :] = torch.tensor(point)
                click_types[i, j] = 1 if j < len(click["fg"]) else 0

        return [points.to(self.model_device), click_types.to(self.model_device)]

    def _get_mask_logits(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        mask_logits: Any = data.get("mask_logits", None)
        if mask_logits is not None:
            mask_logits = mask_logits.to(self.model_device)
        return mask_logits

    def _get_spacing(self, data: Dict[str, Any]) -> np.ndarray:
        return np.array(data["spacing"])

    def _save_mask_logits(self, mask_logits: torch.Tensor) -> None:
        parts: List[str] = self.full_file.split(os.sep)
        mask_logits_path: str = os.sep.join(parts[:-1]) + os.sep + "mask_logits_" + parts[-1]
        np.savez(mask_logits_path, mask_logits=mask_logits.cpu().numpy())

    def _load_model(self) -> torch.nn.Module:
        model: torch.nn.Module = model_registry[self.args.model_type]().to(self.model_device)
        ckpt: Dict[str, Any] = torch.load(
            self.args.model_checkpoint, map_location=self.model_device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model

    def _load_segmenter(self) -> Segmenter:
        segmenter = segmenter_registry[self.args.segmenter_type].load(
            Path(self.args.segmenter_checkpoint), torch.device(self.args.segmenter_device)
        )
        return segmenter

    def _expand_image_embeddings(self, image_embeddings: List[torch.Tensor], batch_dim: int):
        return [im_emb.repeat(batch_dim, 1, 1, 1, 1) for im_emb in image_embeddings]

    def _batched_decoder_inference(
        self,
        image_embeddings: torch.Tensor,
        mask_logits: Optional[torch.Tensor],
        points: Optional[List[torch.Tensor]],
        boxes: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - mask_logits: (batch_size, 1, D, H, W)
            - prompt_embeddings: (batch_size, n_points+2, prompt_embedding_size)
        """
        num_boxes = boxes.shape[0]
        mask_logits_list = []
        prompt_embeddings_list = []

        for i in range(0, num_boxes, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_boxes))
            batch_boxes = boxes[batch_slice]

            batch_image_embeddings = self._expand_image_embeddings(image_embeddings, batch_boxes.shape[0])

            mask_logits_batch, prompt_embeddings = decoder_forward(
                self.model,
                batch_image_embeddings,
                mask_logits=mask_logits[batch_slice] if mask_logits is not None else None,
                points=tuple(p[batch_slice] for p in points) if points is not None else None,
                boxes=batch_boxes,
            )
            mask_logits_list.append(mask_logits_batch.cpu())
            prompt_embeddings_list.append(prompt_embeddings.cpu())

        mask_logits = torch.cat(mask_logits_list, dim=0)
        prompt_embeddings = torch.cat(prompt_embeddings_list, dim=0)

        return mask_logits, prompt_embeddings

    def _log_model_view(self, mask_logits: torch.Tensor, volume: torch.tensor, boxes: torch.Tensor) -> None:
        pred_prob = torch.sigmoid(mask_logits)
        pred_concat = torch.cat((torch.ones_like(mask_logits)[0:1] * 0.5, pred_prob), dim=0)
        pred_long = pred_concat.argmax(dim=0).squeeze(0)
        self.log_predictions_niigz(volume[0, 0], boxes, pred_long.squeeze(), save_dir="work_dir/inference_cropped")

    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        # Most of the data is not in tensor format, so it needs to be converted
        boxes_orig_shape: torch.Tensor = self._get_initial_boxes(
            data
        )  # shape (I, 2, 3) where I is the number of instances in this image
        points: Optional[List[torch.Tensor]] = self._get_points(data)
        mask_logits: Optional[torch.Tensor] = self._get_mask_logits(data)
        spacing: np.ndarray = self._get_spacing(data)

        volume_orig_shape: torch.Tensor = self._transform(data["imgs"]).to(self.model_device)  # shape (1, 1, D, H, W)
        volume, points, boxes = self.coord_handler.forward(volume_orig_shape, spacing, points, boxes_orig_shape)

        # autocast
        # `model` assumes batch dimension
        with safe_autocast(device_type=self.model_device.split(":")[0]):
            # image_embeddings: list[(1, C, D, H, W)]
            image_embeddings, _ = self.model.segresnet(volume)
            # mask_logits: (I, 1, D, H, W)
            mask_logits, prompt_embeddings = self._batched_decoder_inference(
                image_embeddings, mask_logits, points, boxes, batch_size=8
            )

        self._log_model_view(mask_logits, volume, boxes)
        self._save_mask_logits(mask_logits)

        # Still (I, 1, D, H, W) but now in original image space
        mask_logits_orig_shape = self.coord_handler.backward(mask_logits)

        # The Segmenter is responsible for converting a masks of logits to binary segmentations
        # if points is None:
        #     point_coords = None
        #     point_labels = None
        # else:
        #     point_coords = points[0]
        #     point_labels = points[1]

        with safe_autocast(device_type=self.segmenter_device.split(":")[0]):
            device = self.segmenter_device
            padded_prompt_embeddings, prompt_embedding_attention_mask = pad_prompt_embeddings(
                prompt_embeddings, self.n_steps
            )
            td = TensorDict(
                {
                    "mask": mask_logits_orig_shape,
                    "bbox": boxes_orig_shape,
                    "padded_prompt_embeddings": padded_prompt_embeddings,
                    "prompt_embedding_attention_mask": prompt_embedding_attention_mask,
                },
                batch_size=[mask_logits_orig_shape.shape[0]],
                device=device,
            )
            binarized_pred_orig_shape: Integer[torch.Tensor, "image_depth image_height image_width"] = self.segmenter(
                td
            )

        # Log function expects numpy arrays without batch/channel dimension for image and segmentations
        binarized_pred_orig_shape = binarized_pred_orig_shape.cpu().numpy()
        volume_orig_shape = volume_orig_shape[0, 0].cpu().numpy()

        # Log predictions in original image space
        self.log_predictions_niigz(volume_orig_shape, boxes_orig_shape, binarized_pred_orig_shape)

        return binarized_pred_orig_shape

    def log_predictions_niigz(
        self, volume: np.ndarray, boxes: torch.Tensor, binarized_pred: np.ndarray, save_dir: str = "work_dir/inference"
    ) -> None:
        """
        Logs images, bboxes and binarized segmentations to NIfTI files.

        volume: np.ndarray[shape=(D, H, W)]
        boxes: torch.Tensor[shape=(I, 2, 3)]
        binarized_pred: np.ndarray[shape=(D, H, W)]
        """
        os.makedirs(save_dir, exist_ok=True)

        pred_path = os.path.join(save_dir, "pred.nii.gz")
        img_path = os.path.join(save_dir, "img.nii.gz")
        boxes_path = os.path.join(save_dir, "boxes.nii.gz")

        lab: nib.Nifti1Image = nib.Nifti1Image(binarized_pred.astype(np.float32), np.eye(4))
        nib.save(lab, pred_path)

        img_nii: nib.Nifti1Image = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
        nib.save(img_nii, img_path)

        box_volume = np.zeros_like(binarized_pred)
        for i, box in enumerate(boxes, 1):
            box = box.round().cpu().int()
            box_volume[
                box[0, 0] : box[1, 0],
                box[0, 1] : box[1, 1],
                box[0, 2] : box[1, 2],
            ] = i

        lab: nib.Nifti1Image = nib.Nifti1Image(box_volume.astype(np.float32), np.eye(4))
        nib.save(lab, boxes_path)

    def run(self) -> None:
        files: List[str] = [
            f
            for f in os.listdir(self.args.load_path)
            if f.endswith(".npz") and "boxes_" not in f and "mask_logits_" not in f
        ]
        if not files:
            raise ValueError("No input file found in load_path")
        file_name: str = files[0]
        self.full_file: str = os.path.join(self.args.load_path, file_name)

        data: Dict[str, Any] = self.load_data()
        binary_segmentation: Integer[np.ndarray, "image_depth image_height image_width"] = self.predict(data)

        save_file_path: str = os.path.join(self.args.save_path, file_name)
        np.savez(save_file_path, segs=binary_segmentation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict multi-class segmentation of all input images in a folder.")
    parser.add_argument("--load_path", type=Path, help="Folder path to the input image.", default="./inputs")
    parser.add_argument("--save_path", type=Path, help="Folder path to save the predictions.", default="./outputs")
    parser.add_argument(
        "--model_type", type=str, help="Model type to use for prediction.", default="ahus_model_rope_mixed"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="weights/rope_mixed_120_accum/model_latest.pth",
        help="Path to the model weights.",
    )
    parser.add_argument("--model_device", type=str, default="cuda", help="Which device to run the image model on.")
    parser.add_argument(
        "--segmenter_type",
        type=str,
        default="original",
        help="Segmenter type to use for binarizing.",
        choices=list(segmenter_registry.keys()),
    )
    parser.add_argument(
        "--segmenter_checkpoint",
        type=str,
        required=False,  # Some segmenters do not require a checkpoint
        help="Path to saved segmenter.",
    )
    parser.add_argument(
        "--segmenter_device",
        type=str,
        default="cuda",
        help="Which device to run the segmenter on.",
    )
    parser.add_argument("--size_threshold", type=int, default=128**3, help="Size of the input image.")
    parser.add_argument(
        "--n_steps", type=int, default=5, help="How many steps the inference is assumed to be used for."
    )

    args: argparse.Namespace = parser.parse_args()
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
