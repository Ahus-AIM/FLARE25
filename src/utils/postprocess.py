import torch
from torch.nn import functional as F


def trilinear_upsample_threshold(mask_logits: torch.Tensor, image: torch.Tensor, threshold: float) -> torch.Tensor:
    F.interpolate(mask_logits, size=image.shape[-3:], mode="trilinear", align_corners=False)
    return (mask_logits.sigmoid() > threshold).float()
