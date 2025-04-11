import argparse
import os
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import CropForeground
from tensordict import TensorDict

from src.model.build_ahus_model import model_registry
from src.rl.agents import PPOThresholdAgent
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
        self.agent = self._load_agent()
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
        Ensure that the 'boxes' information is present.
          - If already in data, save it to an auxiliary file.
          - Otherwise, load it from the auxiliary file.
        """
        boxes_path: str = self._get_auxiliary_path(self.full_file, "boxes_")
        if "boxes" in data:
            self._save_npz(boxes_path, boxes=data["boxes"])
        else:
            boxes_data: Dict[str, Any] = self._load_npz(boxes_path)
            data["boxes"] = boxes_data["boxes"]
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
        boxes: Any = data["boxes"]
        boxes_tensor: torch.Tensor = torch.zeros((len(boxes), 2, 3), dtype=torch.float32)
        for i, box in enumerate(boxes):
            boxes_tensor[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]])
            boxes_tensor[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]])
        return boxes_tensor.to(self.device)

    def _get_points(self, data: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
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

    def _add_to_logits(
        self, logits: torch.Tensor, box_i: torch.Tensor, box_margin: int = 1, increment: int = 1.0
    ) -> torch.Tensor:
        box = box_i.clone().round().int()
        box[0] = torch.clamp(box[0] - box_margin, 0, logits.shape[-1])
        box[1] = torch.clamp(box[1] + box_margin, 0, logits.shape[-1])
        logits[0, box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]] += increment
        return logits

    def _binarize_output(
        self,
        pred: torch.Tensor,
        boxes: torch.Tensor,
        threshold: float = 0.5,
        ensure_all_present: bool = True,
        max_iter: int = 10,
    ) -> np.ndarray:
        any_vol_zero: bool = True
        counter = 0
        while any_vol_zero and counter < max_iter:
            pred_prob = torch.sigmoid(pred)
            pred_concat = torch.cat((torch.ones_like(pred)[0:1] * threshold, pred_prob), dim=0)
            pred_long = pred_concat.argmax(dim=0).squeeze(0)

            if not ensure_all_present:
                return pred_long.cpu().numpy()

            any_vol_zero = len(torch.unique(pred_long)) != pred.shape[0] + 1

            present = torch.unique(pred_long)
            for i in range(pred.shape[0]):
                if i not in present - 1:
                    pred[i] = self._add_to_logits(pred[i], boxes[i], increment=2**counter)

            counter += 1

        return pred_long.cpu().numpy()

    def _save_mask_logits(self, mask_logits: torch.Tensor) -> None:
        parts: List[str] = self.full_file.split(os.sep)
        mask_logits_path: str = os.sep.join(parts[:-1]) + os.sep + "mask_logits_" + parts[-1]
        np.savez(mask_logits_path, mask_logits=mask_logits.cpu().numpy())

    def _load_model(self) -> torch.nn.Module:
        model: torch.nn.Module = model_registry[self.args.model_type]().to(self.device)
        ckpt: Dict[str, Any] = torch.load(self.args.checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model

    def _load_agent(self) -> PPOThresholdAgent:
        return PPOThresholdAgent.load(self.args.agent_folder_path, torch.device(self.device))

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
        boxes: torch.Tensor = self._get_initial_boxes(data)
        points: Optional[List[torch.Tensor]] = self._get_points(data)
        mask_logits: Optional[torch.Tensor] = self._get_mask_logits(data)
        spacing: np.ndarray = self._get_spacing(data)

        image5D: torch.Tensor = self._transform(data["imgs"]).to(self.device)
        image5D, mask_logits, points, boxes = self.coord_handler.forward(image5D, spacing, mask_logits, points, boxes)
        # autocast
        with torch.autocast(device_type=self.device.split(":")[0]):
            image_embeddings, _ = self.model.segresnet(image5D)
            mask_logits = self._batched_decoder_inference(image_embeddings, mask_logits, points, boxes, batch_size=4)

        self._save_mask_logits(mask_logits)

        self.log_predictions_niigz(mask_logits, image5D, boxes)

        mask_logits_orig_shape = self.coord_handler.backward(mask_logits)

        # Let a RL agent choose the threshold
        # TODO: give the agent more information
        tensordict = TensorDict(
            {"mask": mask_logits_orig_shape}, batch_size=mask_logits_orig_shape.shape[0], device=self.device
        )
        tensordict = self.agent.policy(tensordict)
        # TODO: use one threshold per sample
        threshold: float = tensordict["threshold"][0].item()  # Temporary: use the first threshold for all samples

        binarized_pred: np.ndarray = self._binarize_output(mask_logits_orig_shape, boxes, threshold=threshold)

        return binarized_pred

    def log_predictions_niigz(
        self, mask_logits: np.ndarray, image5D: np.ndarray, boxes: torch.Tensor, save_dir: str = "work_dir/inference"
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        pred_path = os.path.join(save_dir, "pred.nii.gz")
        img_path = os.path.join(save_dir, "img.nii.gz")
        boxes_path = os.path.join(save_dir, "boxes.nii.gz")

        binarized_pred: np.ndarray = self._binarize_output(mask_logits, boxes)
        lab: nib.Nifti1Image = nib.Nifti1Image(binarized_pred.astype(np.float32), np.eye(4))
        nib.save(lab, pred_path)

        img: np.ndarray = image5D[0, 0].cpu().numpy()
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
        prediction: np.ndarray = self.predict(data)

        save_file_path: str = os.path.join(self.args.save_path, file_name)
        np.savez(save_file_path, segs=prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the segmentation of the input images.")
    parser.add_argument("--load_path", type=str, help="Path to the input images.")
    parser.add_argument("--save_path", type=str, help="Path to save the predictions.")
    parser.add_argument(
        "--model_type", type=str, default="ahus_model_rope_mixed", help="Model type to use for prediction."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/weights/5_april/rope_mixed/model_0_step_dice:0.9447498917579651_best.pth",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--agent_folder_path",
        type=str,
        default="./weights/rl/ppo-threshold/apr11",
        help="Path to saved agent.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on.")
    parser.add_argument("--size_threshold", type=int, default=256**3, help="Size of the input image.")

    args: argparse.Namespace = parser.parse_args()
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
