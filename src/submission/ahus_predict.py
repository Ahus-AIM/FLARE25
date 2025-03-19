import argparse
import os
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.model.build_ahus_model import model_registry
from src.utils.decode import decoder_forward

torch.set_grad_enabled(False)


class VolumeTransforms:
    def __init__(self, img_size: int = 128) -> None:
        self.img_size: int = img_size

    @staticmethod
    def _normalize_volume(volume: torch.Tensor) -> torch.Tensor:
        volume = volume.clone()
        volume[volume <= 0] = torch.nan
        positive_volume = volume[~torch.isnan(volume)]
        min_val = positive_volume.min()
        max_val = positive_volume.max()
        volume = (volume - min_val + 1) / (max_val - min_val + 1)
        volume[torch.isnan(volume)] = 0
        return volume

    def _crop_volume(self, volume: torch.Tensor) -> torch.Tensor:
        data_3d: torch.Tensor = volume[0, 0]
        dims = data_3d.shape
        slices = []
        for dim in range(3):
            projection = data_3d.sum(dim=tuple(set(range(3)) - {dim}))
            nonzero = torch.nonzero(projection)
            if nonzero.numel() == 0:
                start, end = 0, dims[dim]
            else:
                start, end = int(nonzero[0].item()), int(nonzero[-1].item()) + 1
            slices.append(slice(start, end))
        cropped = volume[..., slices[0], slices[1], slices[2]]
        self.crop_slices = slices
        return cropped

    def _resample_volume(self, volume: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        dim_z = spacing[2] * volume.shape[2]
        dim_y = spacing[1] * volume.shape[3]
        dim_x = spacing[0] * volume.shape[4]
        max_dim = max(dim_x, dim_y, dim_z)
        self.orig_shape = volume.shape[-3:]
        spacing = spacing / max_dim * self.img_size
        effective_spacing = torch.tensor([spacing[2], spacing[1], spacing[0]])
        target_spacing = torch.tensor([1, 1, 1])
        new_shape = torch.round(torch.tensor(volume.shape[2:]) * effective_spacing / target_spacing).int()
        for i in range(3):
            new_shape[i] = new_shape[i].clamp(min=volume.shape[2 + i], max=torch.tensor(self.img_size))
        self.resampled_shape = new_shape.tolist()
        return F.interpolate(volume, size=self.resampled_shape, mode="trilinear", align_corners=False)

    def _pad_volume(self, volume: torch.Tensor) -> torch.Tensor:
        pad = [0, 0, 0, 0, 0, 0]
        for j in range(3):
            i = 2 - j
            diff = self.img_size - volume.shape[2 + j]
            if diff > 0:
                pad[2 * i + 1] = diff
        self.before_padding_shape = volume.shape[-3:]
        self.padding = tuple(pad)
        return F.pad(volume, self.padding)

    def resample_volume(
        self, image5D: torch.Tensor, spacing: torch.Tensor
    ) -> Optional[torch.Tensor]:  # TODO crop volume as in training script.
        self.orig_shape = image5D.shape[-3:]
        image5D = self._normalize_volume(image5D)
        image5D = self._resample_volume(image5D, spacing)
        image5D = self._pad_volume(image5D)
        return image5D

    def transform_coordinates(
        self, coords: torch.Tensor, original_shape: torch.Tensor, new_shape: torch.Tensor
    ) -> torch.Tensor:
        scale = new_shape.to(torch.float32) / original_shape.to(torch.float32)
        transformed = coords * scale.to(coords.device)
        return transformed

    def forward(
        self,
        image5D: torch.Tensor,
        spacing: np.ndarray,
        mask_logits: Optional[torch.Tensor] = None,
        points: Optional[List[torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
    ):
        image5D = self.resample_volume(image5D, torch.tensor(spacing))
        if points is not None:
            point_coords, point_labels = points
            point_coords = self.transform_coordinates(
                point_coords, torch.tensor(self.orig_shape), torch.tensor(self.before_padding_shape)
            )
            points = [point_coords, point_labels]
        if boxes is not None:
            boxes = self.transform_coordinates(
                boxes, torch.tensor(self.orig_shape), torch.tensor(self.before_padding_shape)
            )
        return image5D, mask_logits, points, boxes

    def backward(self, mask_logits: torch.Tensor):
        mask_logits = F.pad(mask_logits, tuple(-p for p in self.padding))
        mask_logits = F.interpolate(mask_logits, size=self.orig_shape, mode="trilinear")
        # mask_logits = mask_logits[..., self.crop_slices[0], self.crop_slices[1], self.crop_slices[2]]
        return mask_logits


class InferencePipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        self.device: str = args.device
        self.model: torch.nn.Module = self._load_model()
        self.coord_handler: VolumeTransforms = VolumeTransforms(args.img_size)

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
        click_types: torch.Tensor = torch.zeros((len(data["clicks"])), dtype=torch.float32)

        for i, click in enumerate(data["clicks"]):
            all_clicks: List[Any] = click["fg"] + click["bg"]
            for j, point in enumerate(all_clicks):
                points[i, j, :] = torch.tensor(point)
                click_types[i] = 1 if j < len(click["fg"]) else 0

        return [points.to(self.device), click_types.to(self.device)]

    def _get_mask_logits(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        mask_logits: Any = data.get("mask_logits", None)
        if mask_logits is not None:
            mask_logits = torch.tensor(mask_logits).to(self.device)
        return mask_logits

    def _get_spacing(self, data: Dict[str, Any]) -> np.ndarray:
        return np.array(data["spacing"])

    def _binarize_output(self, pred: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        pred = torch.sigmoid(pred)
        pred = torch.cat((torch.ones_like(pred)[0:1] * threshold, pred), dim=0)
        pred = pred.argmax(dim=0).squeeze(0)
        return pred.cpu().numpy()

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
                points=[p[batch_slice] for p in points] if points is not None else None,
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

        image_embeddings, _ = self.model.segresnet(image5D)
        mask_logits = self._batched_decoder_inference(image_embeddings, mask_logits, points, boxes, batch_size=16)

        self._save_mask_logits(mask_logits)

        self.log_predictions_niigz(mask_logits, image5D, boxes)

        mask_logits_orig_shape = self.coord_handler.backward(mask_logits)
        binarized_pred: np.ndarray = self._binarize_output(mask_logits_orig_shape)

        return binarized_pred

    def log_predictions_niigz(
        self, mask_logits: np.ndarray, image5D: np.ndarray, boxes: torch.Tensor, save_dir: str = "work_dir/inference"
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        pred_path = os.path.join(save_dir, "pred.nii.gz")
        img_path = os.path.join(save_dir, "img.nii.gz")
        boxes_path = os.path.join(save_dir, "boxes.nii.gz")

        binarized_pred: np.ndarray = self._binarize_output(mask_logits)
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
    parser.add_argument("--model_type", type=str, default="ahus_model", help="Model type to use for prediction.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/weights/17_mars/sam_model_0_step_dice:0.9522787928581238_best.pth",
        help="Path to the model weights.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on.")
    parser.add_argument("--img_size", type=int, default=128, help="Size of the input image.")

    args: argparse.Namespace = parser.parse_args()
    pipeline: InferencePipeline = InferencePipeline(args)
    pipeline.run()
