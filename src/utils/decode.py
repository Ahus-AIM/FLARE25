from typing import Optional, Tuple

import torch


def decoder_forward(
    model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    mask_logits: Optional[torch.Tensor] = None,
    points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    boxes: Optional[torch.Tensor] = None,
):
    sparse_emb, dense_emb = model.prompt_encoder(points, boxes, mask_logits)
    mask_logits = model.mask_decoder(image_embeddings, model.prompt_encoder.get_dense_pe(), sparse_emb, dense_emb)
    return mask_logits
