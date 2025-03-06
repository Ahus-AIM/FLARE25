from typing import Optional, Tuple

import torch
from torch.nn import functional as F


def decode_batch(
    sam_model: torch.nn.Module,
    image_embedding: torch.Tensor,
    gt3D: torch.Tensor,
    low_res_masks: Optional[torch.Tensor] = None,
    points: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    next_low_res_masks = decode_batch_low_res(
        sam_model=sam_model,
        image_embedding=image_embedding,
        low_res_masks=low_res_masks,
        points=points,
        boxes=boxes,
    )
    next_masks = F.interpolate(next_low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False)
    return next_low_res_masks, next_masks


def decode_batch_low_res(
    sam_model: torch.nn.Module,
    image_embedding: torch.Tensor,
    low_res_masks: Optional[torch.Tensor] = None,
    points: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=low_res_masks,
    )  # type: ignore
    next_low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # type: ignore
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )  # type: ignore
    return next_low_res_masks
