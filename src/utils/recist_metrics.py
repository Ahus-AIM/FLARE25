"""Local per-lesion DSC/NSD evaluation for the FLARE RECIST predictions.

Arrays use ZYX order; spacing comes from the original NPZ in XYZ order.
Missing lesions score zero. Prediction IDs absent from the reference are ignored.
Empty-reference cases are rejected because their scoring policy is unspecified.
"""

import numpy as np

from src.utils.surface_dice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)


def score_case(gt, prediction, spacing_xyz):
    """Score each reference lesion, then average lesions equally within the case.

    NSD uses a 2 mm tolerance and is zero when lesion DSC is below 0.2.
    This is the paper's local protocol, not a claim of private-scorer equivalence.
    """
    if gt.ndim != 3 or gt.shape != prediction.shape:
        raise ValueError("Reference and prediction must have matching ZYX volume shapes")
    for array in (gt, prediction):
        if not np.isfinite(array).all() or (array < 0).any() or not np.equal(array, np.floor(array)).all():
            raise ValueError("Expected finite, nonnegative integer lesion labels")
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Expected three positive XYZ spacing values")
    labels = np.unique(gt[gt > 0])
    if not len(labels):
        raise ValueError("Empty reference: a scoring policy must be explicitly established")

    lesions = []
    for label in labels:
        reference_mask = gt == label
        prediction_mask = prediction == label
        dsc = float(compute_dice_coefficient(reference_mask, prediction_mask))
        nsd = 0.0
        if dsc >= 0.2:
            distances = compute_surface_distances(reference_mask, prediction_mask, spacing[::-1])
            nsd = float(compute_surface_dice_at_tolerance(distances, 2.0))
        lesions.append({"label": int(label), "dsc": dsc, "nsd": nsd})
    means = {metric: float(np.mean([lesion[metric] for lesion in lesions])) for metric in ("dsc", "nsd")}
    return {"n_lesions": len(lesions), "means": means, "lesions": lesions}
