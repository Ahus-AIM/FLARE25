from typing import Tuple

import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from scipy.integrate import cumulative_trapezoid

from src.custom_types import Segmentation
from src.utils.surface_dice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)


def compute_multi_class_dsc(gt: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    dsc = []
    for i in torch.unique(gt)[1:]:  # skip bg
        gt_i = (gt == i).detach().cpu().numpy()
        seg_i = (seg == i).detach().cpu().numpy()
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return torch.tensor(np.mean(dsc)).to(gt.device)


def compute_multi_class_nsd(
    gt: torch.Tensor, seg: torch.Tensor, spacing: torch.Tensor, tolerance: float = 2.0
) -> torch.Tensor:
    nsd = []
    for i in torch.unique(gt)[1:]:  # skip bg
        gt_i = (gt == i).detach().cpu().numpy()
        seg_i = (seg == i).detach().cpu().numpy()
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing.detach().cpu().numpy()
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return torch.tensor(np.mean(nsd)).to(gt.device)


def compute_multi_class_dsc_nsd(
    gt: torch.Tensor, seg: torch.Tensor, spacing: torch.Tensor, tolerance: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gt.shape == seg.shape, "Input tensors must have the same shape"
    assert gt.ndim == 4, "Expected input shape (1, H, W, D)"
    nsd = []
    dsc = []
    for i in torch.unique(gt)[1:]:  # skip bg
        gt_i = (gt[0] == i).detach().cpu().numpy()
        seg_i = (seg[0] == i).detach().cpu().numpy()
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing.detach().cpu().numpy()
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    dsc_tensor = torch.tensor(np.mean(dsc)).to(gt.device)
    nsd_tensor = torch.tensor(np.mean(nsd)).to(gt.device)

    if dsc_tensor < 0.2:
        nsd_tensor = nsd_tensor * 0
    return dsc_tensor, nsd_tensor


# NOTE: This function returns Rewards, but the last dimension is missing
@jaxtyped(typechecker=beartype)
def compute_multi_class_dsc_nsd_batch(
    gt: Segmentation, seg: Segmentation, spacing: torch.Tensor, tolerance: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gt.shape == seg.shape, "Input tensors must have the same shape"
    assert gt.ndim == 5, "Expected input shape (B, 1, H, W, D)"
    dsc = torch.zeros(gt.shape[0], device=gt.device)
    nsd = torch.zeros(gt.shape[0], device=gt.device)
    for i in range(gt.shape[0]):
        dsc[i], nsd[i] = compute_multi_class_dsc_nsd(
            gt[i], seg[i], spacing[i], tolerance
        )
    return dsc, nsd


def get_rewards(
    gt: torch.Tensor,
    seg: torch.Tensor,
    spacing: torch.Tensor,
    tolerance: float = 2.0,
    num_clicks: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert seg.shape[1] == num_clicks + 1, (
        "Expected input shape (B, num_clicks + 1, H, W, D)"
    )
    dscs = torch.zeros((seg.shape[0], num_clicks + 1), device=gt.device)
    nsds = torch.zeros((seg.shape[0], num_clicks + 1), device=gt.device)
    for click in range(num_clicks + 1):
        dsc, nsd = compute_multi_class_dsc_nsd_batch(
            gt, seg[:, click : click + 1], spacing, tolerance
        )
        dscs[:, click] = dsc
        nsds[:, click] = nsd
    dsc_auc = cumulative_trapezoid(dscs.numpy(), axis=1)[:, -1] / (num_clicks)
    nsd_auc = cumulative_trapezoid(nsds.numpy(), axis=1)[:, -1] / (num_clicks)
    dsc_final = dsc
    nsd_final = nsd
    return dsc_auc, nsd_auc, dsc_final, nsd_final


if __name__ == "__main__":
    gt = torch.zeros(4, 1, 16, 16, 16, dtype=torch.long)
    seg = torch.zeros(4, 1, 16, 16, 16, dtype=torch.long)
    spacing = torch.ones(4, 3)
    gt[:, 0, 4:12, 4:12, 4:12] = 2
    seg[:, 0, 4:12, 4:6, 4:12] = 2

    print(compute_multi_class_dsc_nsd_batch(gt, seg, spacing))
    print(get_rewards(gt, seg.repeat(1, 6, 1, 1, 1), spacing))
