from typing import List, Optional, Tuple

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from src.custom_types import ImageEmbedding, Mask, PointLabels, Points
from src.model.modeling import AhusModel


@jaxtyped(typechecker=beartype)
def decoder_forward(
    model: AhusModel,
    image_embeddings: List[ImageEmbedding],
    mask_logits: Mask | None = None,
    points: Optional[Tuple[Points, PointLabels]] = None,
    boxes: Optional[torch.Tensor] = None,
) -> Mask:
    batch_size = image_embeddings[0].shape[0]
    sparse_emb, sparse_emb_pe_term, sparse_emb_pe_factor, dense_emb = model.prompt_encoder(points, boxes, mask_logits)
    mask_logits = model.mask_decoder(
        image_embeddings,
        model.prompt_encoder.get_dense_pe_term(batch_size),
        model.prompt_encoder.get_dense_pe_factor(batch_size),
        sparse_emb,
        sparse_emb_pe_term,
        sparse_emb_pe_factor,
        dense_emb,
    )
    return mask_logits
