"""
This command is expected to take (D, H, W) images from one folder and write (D, H, W) segmentations (integer valued) to a different folder
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Integer
from monai.transforms import CropForeground

from src.model.build_ahus_model import model_registry
from src.submission.segmentation import Segmenter, segmenter_registry
from src.utils.decode import decoder_forward

torch.set_grad_enabled(False)


class VolumeTransforms:
    def __init__(self, size_threshold: int) -> None:
        self.size_threshold = size_threshold
        self.pooling_factors = [1, 1, 1]
        self.crop_slices = None
        self.orig_shape = None
        self.padded_shape = None

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

    def _adaptive_max_pool(self, volume: torch.Tensor) -> torch.Tensor:
        shape = list(volume.shape[2:])  # D, H, W
        self.pooling_factors = [1, 1, 1]
        while volume.numel() > self.size_threshold:
            min_dim = int(np.argmin(shape))
            kernel_size = [2, 2, 2]
            kernel_size[min_dim] = 1
            volume = F.max_pool3d(volume, kernel_size=kernel_size)
            shape = list(volume.shape[2:])
            self.pooling_factors = [self.pooling_factors[i] * kernel_size[i] for i in range(3)]
            print(volume.shape, volume.numel(), self.size_threshold)
        return volume

    def _crop_volume(self, volume: torch.Tensor) -> torch.Tensor:
        cropped = CropForeground(select_fn=lambda x: x > 0, k_divisible=8, allow_smaller=True)(
            volume.squeeze(0)
        ).unsqueeze(0)
        self.crop_slices = []
        for dim in range(3):
            start = int((volume.shape[2 + dim] - cropped.shape[2 + dim]) // 2)
            end = start + cropped.shape[2 + dim]
            self.crop_slices.append(slice(start, end))
        return cropped

    def preprocess_volume(self, image5D: torch.Tensor) -> torch.Tensor:
        image5D = image5D.clone()
        self.orig_shape = image5D.shape[-3:]
        image5D = self._normalize_volume(image5D)

        # Downsample with tracking
        image5D = self._adaptive_max_pool(image5D)

        self.padded_shape = image5D.shape[-3:]  # After pooling but before cropping

        # Crop and track slices
        image5D = self._crop_volume(image5D)
        print("C", image5D.shape)

        # Normalize again post-crop for safety
        image5D = image5D / image5D.max()
        print("D", image5D.shape)
        return image5D

    def transform_coordinates(self, coords: torch.Tensor, direction: str = "forward") -> torch.Tensor:
        coords = coords.clone()
        if direction == "forward":
            for i in range(3):
                coords[..., i] = coords[..., i] / self.pooling_factors[i]
                if self.crop_slices is not None:
                    coords[..., i] -= self.crop_slices[i].start
        return coords

    def forward(
        self,
        image5D: torch.Tensor,
        spacing: np.ndarray = None,
        mask_logits: Optional[torch.Tensor] = None,
        points: Optional[List[torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
    ):
        image5D = self.preprocess_volume(image5D)

        if points is not None:
            point_coords, point_labels = points
            point_coords = self.transform_coordinates(point_coords, direction="forward")
            points = [point_coords, point_labels]
        if boxes is not None:
            boxes = self.transform_coordinates(boxes, direction="forward")

        return image5D, mask_logits, points, boxes

    def backward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        # Uncrop (pad back to pooled size)
        pad_sizes = []
        for dim in range(3):
            cropped_len = mask_logits.shape[2 + dim]
            full_len = self.padded_shape[dim]
            pad_before = self.crop_slices[dim].start
            pad_after = full_len - pad_before - cropped_len
            pad_sizes.extend([pad_before, pad_after])
        pad_sizes = pad_sizes[::-1]  # reverse for torch F.pad
        mask_logits = F.pad(mask_logits, pad_sizes)

        # Upsample back to original size
        return F.interpolate(mask_logits, size=self.orig_shape, mode="trilinear", align_corners=False)


class InferencePipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        self.device: str = args.device
        self.model: torch.nn.Module = self._load_model()
        self.segmenter = self._load_segmenter()
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
        else:  # create a bbox covering the whole image
            image_shape = data["imgs"].shape
            # TODO: check if this is correct
            data["boxes"] = [
                {
                    "z_min": 0,
                    "z_max": image_shape[0],
                    "z_mid_y_min": 0,
                    "z_mid_y_max": image_shape[1],
                    "z_mid_x_min": 0,
                    "z_mid_x_max": image_shape[2],
                }
            ]
            self._save_npz(boxes_path, boxes=data["boxes"])

        return data

    def _handle_mask_logits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        If 'mask_logits' is present in data, override it by loading from its auxiliary file.
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
        image5D: torch.Tensor = torch.tensor(image3D_np).float().unsqueeze(0).unsqueeze(0)
        return image5D

    def _get_initial_boxes(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Converts the boxes from the data dictionary into a tensor format.
        """
        boxes: Any = data["boxes"]
        boxes_tensor: torch.Tensor = torch.zeros((len(boxes), 2, 3), dtype=torch.float32)
        for i, box in enumerate(boxes):
            boxes_tensor[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]])
            boxes_tensor[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]])
        return boxes_tensor.to(self.device)

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

        return [points.to(self.device), click_types.to(self.device)]

    def _get_mask_logits(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        mask_logits: Any = data.get("mask_logits", None)
        if mask_logits is not None:
            mask_logits = mask_logits.to(self.device)
        return mask_logits

    def _get_spacing(self, data: Dict[str, Any]) -> np.ndarray:
        return np.array(data["spacing"])

    def _save_mask_logits(self, mask_logits: torch.Tensor) -> None:
        parts: List[str] = self.full_file.split(os.sep)
        mask_logits_path: str = os.sep.join(parts[:-1]) + os.sep + "mask_logits_" + parts[-1]
        np.savez(mask_logits_path, mask_logits=mask_logits.cpu().numpy())

    def _load_model(self) -> torch.nn.Module:
        model: torch.nn.Module = model_registry[self.args.model_type]().to(self.device)
        ckpt: Dict[str, Any] = torch.load(self.args.model_checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model

    def _load_segmenter(self) -> Segmenter:
        segmenter = segmenter_registry[self.args.segmenter_type].load(self.args.segmenter_checkpoint, self.device)
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
    ) -> np.ndarray:
        num_boxes = boxes.shape[0]
        mask_logits_list = []

        for i in range(0, num_boxes, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_boxes))
            batch_boxes = boxes[batch_slice]

            batch_image_embeddings = self._expand_image_embeddings(image_embeddings, batch_boxes.shape[0])

            mask_logits_batch = decoder_forward(
                self.model,
                batch_image_embeddings,
                mask_logits=mask_logits[batch_slice] if mask_logits is not None else None,
                points=tuple(p[batch_slice] for p in points) if points is not None else None,
                boxes=batch_boxes,
            )
            mask_logits_list.append(mask_logits_batch.cpu())

        mask_logits = torch.cat(mask_logits_list, dim=0)

        return mask_logits

    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        # Most of the data is not in tensor format, so it needs to be converted
        boxes: torch.Tensor = self._get_initial_boxes(
            data
        )  # shape (I, 2, 3) where I is the number of instances in this image
        points: Optional[List[torch.Tensor]] = self._get_points(data)
        mask_logits: Optional[torch.Tensor] = self._get_mask_logits(data)
        spacing: np.ndarray = self._get_spacing(data)

        image5D: torch.Tensor = self._transform(data["imgs"]).to(self.device)  # shape (1, 1, D, H, W)
        image5D, mask_logits, points, boxes = self.coord_handler.forward(image5D, spacing, mask_logits, points, boxes)
        # autocast
        # `model` assumes batch dimension
        with torch.autocast(device_type=self.device.split(":")[0]):
            # image_embeddings: list[(1, C, D, H, W)]
            image_embeddings, _ = self.model.segresnet(image5D)
            # mask_logits: (I, 1, D, H, W)
            mask_logits = self._batched_decoder_inference(image_embeddings, mask_logits, points, boxes, batch_size=4)

        self._save_mask_logits(mask_logits)

        # Still (I, 1, D, H, W) but now in original image space
        mask_logits_orig_shape = self.coord_handler.backward(mask_logits)

        # The Segmenter is responsible for converting a masks of logits to binary segmentations
        if points is None:
            point_coords = None
            point_labels = None
        else:
            point_coords = points[0]
            point_labels = points[1]
        binarized_pred: Integer[torch.Tensor, "image_depth image_height image_width"] = self.segmenter(
            logits=mask_logits_orig_shape,
            image=image5D,
            bbox=boxes,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        # Log function expects numpy arrays for image and segmentations
        binarized_pred_np = binarized_pred.cpu().numpy()
        image5D_np = image5D.cpu().numpy()

        # Log predictions
        self.log_predictions_niigz(image5D_np, boxes, binarized_pred_np)

        return binarized_pred_np

    def log_predictions_niigz(
        self, image5D: np.ndarray, boxes: torch.Tensor, binarized_pred: np.ndarray, save_dir: str = "work_dir/inference"
    ) -> None:
        """
        Logs images, bboxes and binarized segmentations to NIfTI files.
        """
        os.makedirs(save_dir, exist_ok=True)

        pred_path = os.path.join(save_dir, "pred.nii.gz")
        img_path = os.path.join(save_dir, "img.nii.gz")
        boxes_path = os.path.join(save_dir, "boxes.nii.gz")

        lab: nib.Nifti1Image = nib.Nifti1Image(binarized_pred.astype(np.float32), np.eye(4))
        nib.save(lab, pred_path)

        # Remove batch and channel dimensions
        img: np.ndarray = image5D[0, 0]

        img_nii: nib.Nifti1Image = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
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
    parser.add_argument("--load_path", type=str, help="Folder path to the input image.")
    parser.add_argument("--save_path", type=str, help="Folder path to save the predictions.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to use for prediction.")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--segmenter_type",
        type=str,
        required=True,
        help="Segmenter type to use for binarizing.",
        choices=list(segmenter_registry.keys()),
    )
    parser.add_argument(
        "--segmenter_checkpoint",
        type=str,
        required=False,  # Some segmenters do not require a checkpoint
        help="Path to saved segmenter.",
    )
    parser.add_argument("--device", type=str, required=True, help="Device to run the inference on.")
    parser.add_argument("--size_threshold", type=int, required=True, help="Size of the input image.")

    args: argparse.Namespace = parser.parse_args()
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
