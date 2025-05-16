import torch
from jaxtyping import Bool
from torch import Tensor

from src.custom_types import MulticlassSegmentation


def batched_binary_segmentation_to_multiclass_segmentation(
    segmentation: Bool[Tensor, "I D H W"],
) -> MulticlassSegmentation:
    """
    Convert a batched binary segmentation to a multiclass segmentation.
    The input is a bool tensor of shape (I, D, H, W) and the output is an integer tensor of shape (D, H, W).
    """
    return torch.stack(
        (
            torch.zeros((1,) + segmentation.shape[1:], dtype=torch.uint8, device=segmentation.device),
            segmentation.to(torch.uint8),
        ),
        dim=0,
    ).argmax(dim=0)
