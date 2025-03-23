from typing import List, Tuple

import cupy as cp
import torch
from beartype import beartype
from cucim.core.operations import morphology
from jaxtyping import jaxtyped


@jaxtyped(typechecker=beartype)
def interact(prediction: torch.Tensor, gt_semantic_seg: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
    # Ensure tensors are Long and on GPU
    assert prediction.dtype == torch.long and gt_semantic_seg.dtype == torch.long, "Inputs must be LongTensors"
    assert prediction.shape == gt_semantic_seg.shape, "Input tensors must have the same shape"
    assert prediction.ndim == 5, "Expected input shape (B, 1, H, W, D)"

    device = prediction.device
    batch_size = gt_semantic_seg.shape[0]

    # Compute error mask
    to_points_mask = (prediction != gt_semantic_seg).squeeze(1)  # Shape: (B, H, W, D)

    batch_points, batch_labels = [], []

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

            batch_points.append(torch.tensor([center], dtype=torch.long, device=device).unsqueeze(0))
            batch_labels.append(label)
        else:
            print(f"[Batch item {i}] No error connected components found. Prediction is perfect! No clicks added.")

    return batch_points, batch_labels


def compute_largest_error_point(error_mask: torch.Tensor) -> Tuple[int, int, int]:
    error_mask_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(error_mask))
    edt_cp = morphology.distance_transform_edt(error_mask_cp)
    center = cp.unravel_index(cp.argmax(edt_cp), edt_cp.shape)

    return (int(center[0]), int(center[1]), int(center[2]))
