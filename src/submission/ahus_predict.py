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
from jaxtyping import Float, Integer

from src.custom_types import BatchedImageLogits, BatchedPointCoords, BatchedPointLabels, Boxes, Image
from src.model.registry import model_registry

# This import is needed for Agent.load to recognize subclasses
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent  # noqa: F401
from src.rl.utils import pad_prompt_embeddings
from src.submission.segmentation import Segmenter, segmenter_registry
from src.transform.transform import VolumeTransforms  # noqa: E402, F401
from src.utils.decode import decoder_forward


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
        self.debug: bool = args.debug

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
            data["boxes"] = None
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
        # data = self._handle_mask_logits(data)
        # data = self._handle_boxes(data)
        return data

    # -------------------- Inference Helpers -------------------- #
    @staticmethod
    def _transform(image3D_np: Float[np.ndarray, "D H W"]) -> Image:
        """Returns a float image tensor of shape (D, H, W) with values in [0, 255]"""
        volume = torch.tensor(image3D_np).float()
        return volume

    @staticmethod
    def _get_initial_boxes(data: Dict[str, Any]) -> Boxes:
        """
        Converts the boxes from the data dictionary into a tensor format.
        """
        boxes: Any = data["boxes"]
        if boxes is None or boxes.size < 1 or str(boxes) == "None":
            points = data.get("clicks", None)
            if points is not None and len(points) > 0:
                boxes = []
                for instance in points:
                    foreground_points = instance.get("fg", [])
                    first_foreground_point = foreground_points[0] if foreground_points else None
                    if first_foreground_point is not None:
                        # Create a box around the first foreground point
                        boxes.append(
                            {
                                "z_min": max(first_foreground_point[0] - 16, 0),
                                "z_max": first_foreground_point[0] + 16,
                                "z_mid_y_min": max(first_foreground_point[1] - 16, 0),
                                "z_mid_y_max": first_foreground_point[1] + 16,
                                "z_mid_x_min": max(first_foreground_point[2] - 16, 0),
                                "z_mid_x_max": first_foreground_point[2] + 16,
                            }
                        )
                print(points)
            else:
                return None
        boxes_tensor: torch.Tensor = torch.zeros((len(boxes), 2, 3), dtype=torch.float32)
        for i, box in enumerate(boxes):
            boxes_tensor[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]])
            boxes_tensor[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]])
        print(boxes)
        return boxes_tensor

    @staticmethod
    def _get_diameter_points(data: Dict[str, Any]) -> Boxes:
        diameter_points = []
        lines = data["recist"]
        for i in np.unique(lines):
            if i == 0:
                continue
            line = np.argwhere(lines == i)
            p1 = line[0]
            p2 = line[-1]
            diameter_points.append((p1, p2))

        diameter_points = torch.tensor(diameter_points, dtype=torch.float32)
        boxes_tensor: torch.Tensor = torch.zeros((len(diameter_points), 2, 3), dtype=torch.float32)

        for i, (p1, p2) in enumerate(diameter_points):
            boxes_tensor[i, 0, :] = torch.tensor(p1)
            boxes_tensor[i, 1, :] = torch.tensor(p2)

        return diameter_points

    def _get_points(self, data: Dict[str, Any]) -> tuple[BatchedPointCoords, BatchedPointLabels] | None:
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

    def _get_image_logits(self, data: Dict[str, Any]) -> BatchedImageLogits | None:
        image_logits: Any = data.get("mask_logits", None)
        if image_logits is not None:
            image_logits = image_logits.to(self.model_device)
        return image_logits

    def _get_spacing(self, data: Dict[str, Any]) -> Float[np.ndarray, "3"]:
        return np.array(data["spacing"])

    def _save_image_logits(self, mask_logits: BatchedImageLogits) -> None:
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

    def _get_num_instances(self, points, boxes) -> int:
        if boxes is not None and boxes.numel() > 2:
            return boxes.shape[0]
        elif points is not None and points[0] is not None and points[0].numel() > 0:
            return points[0].shape[0]
        else:
            return 1

    def _batched_decoder_inference(
        self,
        image_embeddings: torch.Tensor,
        mask_logits: Optional[torch.Tensor],
        points: Optional[tuple[BatchedPointCoords, BatchedPointLabels]],
        boxes: Boxes,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - mask_logits: (batch_size, 1, D, H, W)
            - prompt_embeddings: (batch_size, n_points+2, prompt_embedding_size)
        """
        mask_logits_list = []
        prompt_embeddings_list = []

        num_instances = self._get_num_instances(points, boxes)

        for i in range(0, num_instances, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_instances))
            batch_num_instances = min(batch_size, num_instances - i)

            batch_boxes = None if boxes is None else boxes[batch_slice]

            batch_image_embeddings = self._expand_image_embeddings(image_embeddings, batch_num_instances)

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
            volume.cpu().numpy(),
            boxes,
            pred_long.squeeze(),
            save_dir="work_dir/inference_cropped",
        )

    @torch.no_grad()
    def predict(self, data: dict[str, Any]) -> np.ndarray:
        """Return a multiclass segmentation."""

        # Most of the data is not in tensor format, so it needs to be converted
        # boxes: Boxes = self._get_initial_boxes(data)
        boxes: Boxes = self._get_diameter_points(data)
        if boxes is not None:
            boxes = boxes.to(self.model_device)
        points = self._get_points(data)

        # Image logits can be saved from a previous step and are saved in a downsampled format
        prev_downsampled_image_logits = self._get_image_logits(data)

        unnormalized_volume = self._transform(data["imgs"]).to(self.model_device)

        # Downsample coordinates
        downsampled_volume, downsampled_boxes, downsampled_point_coords = self.coord_handler.forward(
            unnormalized_volume,
            boxes,
            points[0] if points is not None else None,
        )

        if points is None:
            downsampled_points = None
        else:
            assert downsampled_point_coords is not None  # for static type checking
            point_labels = points[1]
            downsampled_point_labels = point_labels  # labels are not coordinate-dependent
            downsampled_points = (downsampled_point_coords, downsampled_point_labels)

        # autocast
        # `model` assumes batch dimension
        with safe_autocast(device_type=self.model_device.split(":")[0]):
            # Batched image embeddings (1, C, D, H, W)
            image_embeddings = self.model.segresnet(downsampled_volume.unsqueeze(0).unsqueeze(0))
            # Multiclass mask_logits: (I, 1, D, H, W)
            downsampled_image_logits, prompt_embeddings = self._batched_decoder_inference(
                image_embeddings,
                prev_downsampled_image_logits.unsqueeze(1) if prev_downsampled_image_logits is not None else None,
                downsampled_points,
                downsampled_boxes,
                batch_size=2,
            )
            downsampled_image_logits = downsampled_image_logits.squeeze(1)  # (I, D, H, W)

        if self.debug:
            self._log_model_view(downsampled_image_logits, downsampled_volume, downsampled_boxes)
        # self._save_image_logits(downsampled_image_logits)

        # (n_instances, D, H, W) and now in original image space
        image_logits: BatchedImageLogits = self.coord_handler.backward(downsampled_image_logits)

        # with safe_autocast(device_type=self.segmenter_device.split(":")[0]):
        with nullcontext():
            padded_prompt_embeddings, prompt_embedding_attention_mask = pad_prompt_embeddings(
                prompt_embeddings, self.n_clicks
            )
            multiclass_segmentation = self.segmenter(
                image_logits.to(self.segmenter_device),
                downsampled_image_logits.to(self.segmenter_device),
                boxes.to(self.segmenter_device) if boxes is not None else None,
                downsampled_boxes.to(self.segmenter_device) if downsampled_boxes is not None else None,
                None,
                None,
                None,
                padded_prompt_embeddings.to(self.segmenter_device),
                prompt_embedding_attention_mask.to(self.segmenter_device),
            )

        if self.debug:
            # Log predictions in original image space
            self.log_predictions_niigz(
                unnormalized_volume.cpu().numpy(),
                boxes,
                multiclass_segmentation.cpu().numpy(),
            )

        return multiclass_segmentation.cpu().numpy()

    @staticmethod
    def log_predictions_niigz(
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

        os.makedirs(os.path.dirname(pred_path), exist_ok=True)

        lab: nib.Nifti1Image = nib.Nifti1Image(multiclass_segmentation.astype(np.float32), np.eye(4))
        nib.save(lab, pred_path)

        img_nii: nib.Nifti1Image = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
        nib.save(img_nii, img_path)

        box_volume = np.zeros_like(multiclass_segmentation)
        if boxes is not None:
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
        torch.set_grad_enabled(False)
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

        import nibabel as nib

        affine = np.eye(4)
        nii_img = nib.Nifti1Image(binary_segmentation.astype(np.float32), affine)
        nib.save(nii_img, f"{self.args.save_path}/{file_name}.nii.gz")


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
        default=128 * 128 * 128,
        help="Size of the input image.",
    )
    parser.add_argument(
        "--n_clicks",
        type=int,
        default=5,
        help="How many steps the inference is assumed to be used for.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Enable debug logging.",
    )

    args: argparse.Namespace = parser.parse_args()
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
