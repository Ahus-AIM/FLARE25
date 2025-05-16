import numpy as np
import torch
from cucim.core.operations import morphology
from scipy.ndimage import distance_transform_edt

from src.custom_types import BatchedChannelSegmentation, BatchedPointCoords, BatchedPointLabels


def interact(
    prediction: BatchedChannelSegmentation, gt_semantic_seg: BatchedChannelSegmentation
) -> tuple[list[BatchedPointCoords], list[BatchedPointLabels]]:
    """
    Get clicks using the same method as in challenge evaluation.

    Arguments:
        prediction (torch.Tensor): Predicted segmentation mask (LongTensor of 0s and 1s) with shape (B, 1, H, W, D).
        gt_semantic_seg (torch.Tensor): Ground truth segmentation mask (LongTensor of 0s and 1s) with the same shape.

    Returns:
        tuple: (batch_points, batch_labels), where:
            - batch_points: List of tensors, each containing a single point (1, 1, 3).
            - batch_labels: List of tensors, each containing a single label (1, 1).
    """
    # Ensure tensors are Long
    assert prediction.dtype == torch.long and gt_semantic_seg.dtype == torch.long, "Inputs must be LongTensors"
    assert prediction.shape == gt_semantic_seg.shape, "Input tensors must have the same shape"
    assert prediction.ndim == 5, "Expected input shape (B, 1, H, W, D)"

    device = prediction.device
    batch_size = gt_semantic_seg.shape[0]

    # Compute error mask
    to_points_mask = (prediction != gt_semantic_seg).squeeze(1)  # Shape: (B, H, W, D)

    batch_points, batch_labels = [], []
    try:
        for i in range(batch_size):
            error_mask = to_points_mask[i]

            if error_mask.sum() > 0:
                center = compute_largest_error_point(error_mask)

                # Place the click: background click for oversegmentation, foreground for undersegmentation
                if gt_semantic_seg[i, 0, center[0], center[1], center[2]] == 0:  # Oversegmentation
                    assert prediction[i, 0, center[0], center[1], center[2]] == 1, "Error in click placement"
                    label = torch.tensor([[0]], device=device)  # Background label
                else:  # Undersegmentation
                    assert prediction[i, 0, center[0], center[1], center[2]] == 0, "Error in click placement"
                    label = torch.tensor([[1]], device=device)  # Foreground label

                batch_points.append(torch.tensor([center], dtype=torch.float, device=device).unsqueeze(0))
                batch_labels.append(label)
            else:
                print(f"[Batch item {i}] No error connected components found. Prediction is perfect! No clicks added.")
    except Exception as e:
        print(f"Error in interact function. Returning empty points lists.\n{e}")
        batch_points, batch_labels = [], []

    return batch_points, batch_labels


def compute_largest_error_point(error_mask: torch.Tensor) -> tuple[int, int, int]:
    if error_mask.device.type == "cuda":
        import cupy as cp

        error_mask_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(error_mask))
        edt_cp = morphology.distance_transform_edt(error_mask_cp)
        center = cp.unravel_index(cp.argmax(edt_cp), edt_cp.shape)
    else:
        error_mask_np = error_mask.cpu().numpy()
        edt_np = distance_transform_edt(error_mask_np)
        center = np.unravel_index(np.argmax(edt_np), edt_np.shape)
    return (int(center[0]), int(center[1]), int(center[2]))
