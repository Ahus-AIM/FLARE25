import torch

from src.custom_types import MulticlassSegmentation
from src.utils.surface_dice import (
    compute_multi_class_dsc,
    compute_multi_class_nsd,
)


def compute_multi_class_dsc_nsd(
    gt: MulticlassSegmentation,
    seg: MulticlassSegmentation,
    spacing: torch.Tensor,
    tolerance: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert gt.shape == seg.shape, f"Input tensors must have the same shape, instead got {gt.shape} and {seg.shape}"
    assert gt.ndim == 3, "Expected input shape (H, W, D)"
    dsc = compute_multi_class_dsc(gt.cpu().numpy(), seg.cpu().numpy())

    if dsc > 0.2:
        nsd = compute_multi_class_nsd(gt.cpu().numpy(), seg.cpu().numpy(), spacing.cpu().numpy(), tolerance)
    else:
        nsd = 0.0
    return torch.tensor(dsc, dtype=torch.float32, device=gt.device), torch.tensor(
        nsd, dtype=torch.float32, device=gt.device
    )
