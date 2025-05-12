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
from jaxtyping import Float, Integer
from monai.transforms.croppad.array import DivisiblePad
from tensordict import TensorDict

from src.custom_types import (
    Boxes,
    Image,
)
from src.model.registry import model_registry
from src.rl.utils import pad_prompt_embeddings
from src.submission.segmentation import Segmenter, segmenter_registry
from src.utils.decode import decoder_forward

torch.set_grad_enabled(False)

# This import is needed for Agent.load to recognize subclasses
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent  # noqa: E402, F401


class VolumeTransforms:
    def __init__(self, size_threshold: int) -> None:
        self.size_threshold: int = size_threshold
        self.pooling_factors: tuple[int, ...] = (1, 1, 1)
        self.crop_slices: tuple[slice, ...] | None = None  # set in forward and used in backward
        self.pad_values: tuple[int, ...] | None = None
        self.orig_shape: tuple[int, ...] | None = None
        self.cropped_shape: tuple[int, ...] | None = None
        self.padder = DivisiblePad(k=8, method="end", value=0)

    @staticmethod
    def _normalize_volume(volume: Image) -> Image:
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

    def _adaptive_max_pool(
        self,
        volume: Image,
        boxes: Boxes,
        point_coords: MulticlassPointCoords | None,
    ) -> tuple[Image, Boxes, MulticlassPointCoords | None]:
        shape = volume.shape  # D, H, W
        self.pooling_factors = (1, 1, 1)

        # F.max_pool3d expects channel dimension
        volume = volume.unsqueeze(0)

        while volume.numel() > self.size_threshold:
            min_dim = int(np.argmin(shape))
            kernel_size = tuple(1 if i == min_dim else 2 for i in range(3))
            volume = F.max_pool3d(volume, kernel_size=kernel_size)
            shape = volume.shape
            self.pooling_factors = tuple(self.pooling_factors[i] * kernel_size[i] for i in range(3))
        # Remove channel dimension
        volume = volume.squeeze(0)

        # TODO: check this
        boxes = boxes / torch.tensor(self.pooling_factors, device=boxes.device).view(1, 3)
        if point_coords is not None:
            point_coords = point_coords / torch.tensor(self.pooling_factors, device=point_coords.device).view(1, 3)

        return volume, boxes, point_coords

    def _crop(
        self, volume: Image, boxes: Boxes, point_coords: MulticlassPointCoords | None
    ) -> tuple[Image, Boxes, MulticlassPointCoords | None]:
        """Return cropped versions with coordinates restricted to the bounding boxes, with a margin of 16 pixels."""
        crop_margin = 16

        xmin = int(boxes[:, :, 0].min().item())
        ymin = int(boxes[:, :, 1].min().item())
        zmin = int(boxes[:, :, 2].min().item())
        xmax = int(boxes[:, :, 0].max().item())
        ymax = int(boxes[:, :, 1].max().item())
        zmax = int(boxes[:, :, 2].max().item())

        # Make sure margin is respected
        xmin = max(0, xmin - crop_margin)
        ymin = max(0, ymin - crop_margin)
        zmin = max(0, zmin - crop_margin)
        xmax = min(volume.shape[0], xmax + crop_margin)
        ymax = min(volume.shape[1], ymax + crop_margin)
        zmax = min(volume.shape[2], zmax + crop_margin)

        cropped_volume = volume[xmin:xmax, ymin:ymax, zmin:zmax].clone()
        self.cropped_shape = cropped_volume.shape
        self.crop_slices = tuple([slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)])

        cropped_boxes = boxes.clone()
        cropped_boxes[..., 0] = boxes[..., 0] - self.crop_slices[0].start
        cropped_boxes[..., 1] = boxes[..., 1] - self.crop_slices[1].start
        cropped_boxes[..., 2] = boxes[..., 2] - self.crop_slices[2].start

        if point_coords is not None:
            cropped_point_coords = point_coords.clone()
            cropped_point_coords[0][..., 0] = point_coords[0][..., 0] - self.crop_slices[0].start
            cropped_point_coords[0][..., 1] = point_coords[0][..., 1] - self.crop_slices[1].start
            cropped_point_coords[0][..., 2] = point_coords[0][..., 2] - self.crop_slices[2].start
        else:
            cropped_point_coords = None

        return cropped_volume, cropped_boxes, cropped_point_coords

    def _pad_divisible(self, volume: Image, k: int = 8) -> Image:
        shape_before = volume.shape
        # DivisiblePad expects a 4D tensor, channel first
        padded_volume: Image = self.padder(volume.unsqueeze(0)).squeeze(0)
        shape_after = padded_volume.shape
        self.pad_values = tuple(shape_after[i] - shape_before[i] for i in range(3))
        return padded_volume

    def preprocess_volume(
        self,
        volume: Image,
        boxes: Boxes,
        point_coords: Optional[MulticlassPointCoords] = None,
    ) -> tuple[
        Image,
        Boxes,
        Optional[MulticlassPointCoords],
    ]:
        self.orig_shape = volume.shape

        # Normalize
        volume = self._normalize_volume(volume)

        volume, boxes, point_coords = self._crop(volume, boxes, point_coords)

        # Downsample with adaptive max pooling
        volume, boxes, point_coords = self._adaptive_max_pool(volume, boxes, point_coords)

        # Pad to divisible size
        volume = self._pad_divisible(volume)

        return volume, boxes, point_coords

    def forward(
        self,
        volume: Image,
        boxes: Boxes,
        point_coords: Optional[MulticlassPointCoords] = None,
    ) -> tuple[
        Image,
        Boxes,
        Optional[MulticlassPointCoords],
    ]:
        volume, boxes, point_coords = self.preprocess_volume(volume, boxes, point_coords)

        return volume, boxes, point_coords

    def backward(self, mask_logits: MulticlassImageLogits) -> MulticlassImageLogits:
        assert self.pad_values is not None, "pad_values must be set before calling backward"
        assert self.orig_shape is not None, "orig_shape must be set before calling backward"
        assert self.crop_slices is not None, "crop_slices must be set before calling backward"
        assert self.cropped_shape is not None, "cropped_shape must be set before calling backward"

        if self.pad_values[0] > 0:
            mask_logits = mask_logits[:, : -self.pad_values[0], :, :]
        if self.pad_values[1] > 0:
            mask_logits = mask_logits[:, :, : -self.pad_values[1], :]
        if self.pad_values[2] > 0:
            mask_logits = mask_logits[:, :, :, : -self.pad_values[2]]

        # F.interpolate expects batch dimension
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=self.cropped_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        pad_sizes = []
        for dim in range(3):
            pad_before = self.crop_slices[dim].start
            pad_after = self.orig_shape[dim] - self.crop_slices[dim].stop
            pad_sizes.extend([pad_after, pad_before])  # NOTE
        pad_sizes = pad_sizes[::-1]  # reverse for torch F.pad

        mask_logits = F.pad(mask_logits, pad_sizes)

        # Upsample back to original size
        # TODO: should not be necessary?
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=self.orig_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return mask_logits


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
        self.n_clicks: int = args.n_clicks
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
    def _transform(self, image3D_np: Float[np.ndarray, "D H W"]) -> Image:
        """Returns a float image tensor of shape (D, H, W) with values in [0, 255]"""
        volume = torch.tensor(image3D_np).float()
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

    def _get_points(self, data: Dict[str, Any]) -> tuple[MulticlassPointCoords, MulticlassPointLabels] | None:
        """
        Converts the points from the data dictionary into a tensor format.
        """
        if "clicks" not in data:
            return None

        num_clicks: int = len(data["clicks"][0]["fg"]) + len(data["clicks"][0]["bg"])
        point_coords: torch.Tensor = torch.zeros((len(data["clicks"]), num_clicks, 3), dtype=torch.float32)
        point_labels: torch.Tensor = torch.zeros((len(data["clicks"]), num_clicks), dtype=torch.long)

        for i, click in enumerate(data["clicks"]):
            all_clicks: List[Any] = click["fg"] + click["bg"]
            for j, point in enumerate(all_clicks):
                point_coords[i, j, :] = torch.tensor(point)
                point_labels[i, j] = 1 if j < len(click["fg"]) else 0

        return (point_coords.to(self.model_device), point_labels.to(self.model_device))

    def _get_mask_logits(self, data: Dict[str, Any]) -> MulticlassImageLogits | None:
        mask_logits: Any = data.get("mask_logits", None)
        if mask_logits is not None:
            mask_logits = mask_logits.to(self.model_device)
        return mask_logits

    def _get_spacing(self, data: Dict[str, Any]) -> Float[np.ndarray, "3"]:
        return np.array(data["spacing"])

    def _save_mask_logits(self, mask_logits: MulticlassImageLogits) -> None:
        parts: List[str] = self.full_file.split(os.sep)
        mask_logits_path: str = os.sep.join(parts[:-1]) + os.sep + "mask_logits_" + parts[-1]
        np.savez(mask_logits_path, mask_logits=mask_logits.cpu().numpy())

    def _load_model(self) -> torch.nn.Module:
        model: torch.nn.Module = model_registry[self.args.model_type]().to(self.model_device)
        ckpt: Dict[str, Any] = torch.load(
            self.args.model_checkpoint,
            map_location=self.model_device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model

    def _load_segmenter(self) -> Segmenter:
        segmenter = segmenter_registry[self.args.segmenter_type].load(
            Path(self.args.segmenter_checkpoint) if self.args.segmenter_checkpoint is not None else None,
            torch.device(self.args.segmenter_device),
        )
        return segmenter

    def _expand_image_embeddings(self, image_embeddings: List[torch.Tensor], batch_dim: int):
        return [im_emb.repeat(batch_dim, 1, 1, 1, 1) for im_emb in image_embeddings]

    def _batched_decoder_inference(
        self,
        image_embeddings: torch.Tensor,
        mask_logits: Optional[torch.Tensor],
        points: Optional[tuple[MulticlassPointCoords, MulticlassPointLabels]],
        boxes: Boxes,
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

    def _log_model_view(self, mask_logits: torch.Tensor, volume: torch.Tensor, boxes: torch.Tensor) -> None:
        pred_prob = torch.sigmoid(mask_logits)
        pred_concat = torch.cat((torch.ones_like(mask_logits)[0:1] * 0.5, pred_prob), dim=0)
        pred_long = pred_concat.argmax(dim=0).squeeze(0)
        self.log_predictions_niigz(
            volume[0, 0].cpu().numpy(),
            boxes,
            pred_long.squeeze(),
            save_dir="work_dir/inference_cropped",
        )

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        """Return a multiclass segmentation."""
        # Most of the data is not in tensor format, so it needs to be converted
        boxes_orig_shape: Boxes = self._get_initial_boxes(
            data
        )  # shape (I, 2, 3) where I is the number of instances in this image
        points_orig_shape = self._get_points(data)
        mask_logits = self._get_mask_logits(data)
        # spacing = self._get_spacing(data)

        unnormalized_volume_orig_shape = self._transform(data["imgs"]).to(self.model_device)
        volume, boxes, point_coords = self.coord_handler.forward(
            unnormalized_volume_orig_shape,
            boxes_orig_shape,
            points_orig_shape[0] if points_orig_shape is not None else None,
        )

        if points_orig_shape is None:
            points = None
        else:
            assert point_coords is not None
            points = (point_coords, points_orig_shape[1])

        # autocast
        # `model` assumes batch dimension
        with safe_autocast(device_type=self.model_device.split(":")[0]):
            # Batched image embeddings (1, C, D, H, W)
            image_embeddings, _ = self.model.segresnet(volume.unsqueeze(0).unsqueeze(0))
            # Multiclass mask_logits: (I, 1, D, H, W)
            mask_logits, prompt_embeddings = self._batched_decoder_inference(
                image_embeddings, mask_logits, points, boxes, batch_size=8
            )

        self._log_model_view(mask_logits, volume, boxes)
        self._save_mask_logits(mask_logits)

        # (n_instances, D, H, W) and now in original image space
        mask_logits_orig_shape = self.coord_handler.backward(mask_logits.squeeze(1))

        with safe_autocast(device_type=self.segmenter_device.split(":")[0]):
            device = self.segmenter_device
            padded_prompt_embeddings, prompt_embedding_attention_mask = pad_prompt_embeddings(
                prompt_embeddings, self.n_clicks
            )
            td = TensorDict(
                {
                    "multiclass_image_logits": mask_logits_orig_shape,
                    "boxes": boxes_orig_shape,
                    "multiclass_padded_prompt_embeddings": padded_prompt_embeddings,
                    "multiclass_prompt_embedding_attention_mask": prompt_embedding_attention_mask,
                },
                device=device,
            )
            multiclass_segmentation = self.segmenter(td)

        # Log predictions in original image space
        self.log_predictions_niigz(
            unnormalized_volume_orig_shape.cpu().numpy(), boxes_orig_shape, multiclass_segmentation.cpu().numpy()
        )

        return multiclass_segmentation.cpu().numpy()

    def log_predictions_niigz(
        self,
        volume: np.ndarray,
        boxes: torch.Tensor,
        multiclass_segmentation: np.ndarray,
        save_dir: str = "work_dir/inference",
    ) -> None:
        """
        Logs images, bboxes and binarized segmentations to NIfTI files.

        volume: np.ndarray[shape=(D, H, W)]
        boxes: torch.Tensor[shape=(I, 2, 3)]
        multiclass_segmentation: np.ndarray[shape=(D, H, W)]
        """
        os.makedirs(save_dir, exist_ok=True)

        pred_path = os.path.join(save_dir, "pred.nii.gz")
        img_path = os.path.join(save_dir, "img.nii.gz")
        boxes_path = os.path.join(save_dir, "boxes.nii.gz")

        lab: nib.Nifti1Image = nib.Nifti1Image(multiclass_segmentation.astype(np.float32), np.eye(4))
        nib.save(lab, pred_path)

        img_nii: nib.Nifti1Image = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
        nib.save(img_nii, img_path)

        box_volume = np.zeros_like(multiclass_segmentation)
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
    parser.add_argument(
        "--load_path",
        type=Path,
        help="Folder path to the input image.",
        default="./inputs",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Folder path to save the predictions.",
        default="./outputs",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type to use for prediction.",
        default="ahus_model_rope_mixed",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="weights/rope_mixed_120_accum/model_latest.pth",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        help="Which device to run the image model on.",
    )
    parser.add_argument(
        "--segmenter_type",
        type=str,
        default="original",
        help="Segmenter type to use for binarizing.",
        choices=list(segmenter_registry.keys()),
    )
    parser.add_argument(
        "--segmenter_checkpoint",
        type=Path,
        required=False,  # Some segmenters do not require a checkpoint
        help="Path to saved segmenter.",
    )
    parser.add_argument(
        "--segmenter_device",
        type=str,
        default="cuda",
        help="Which device to run the segmenter on.",
    )
    parser.add_argument(
        "--size_threshold",
        type=int,
        default=256 * 128 * 128,
        help="Size of the input image.",
    )
    parser.add_argument(
        "--n_clicks",
        type=int,
        default=5,
        help="How many steps the inference is assumed to be used for.",
    )

    args: argparse.Namespace = parser.parse_args()
    print(args.segmenter_checkpoint, type(args.segmenter_checkpoint))
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
