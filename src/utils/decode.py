from src.custom_types import (
    BatchedBox,
    BatchedImageEmbedding,
    BatchedImageLogits,
    BatchedPointCoords,
    BatchedPointLabels,
    BatchedPromptEmbeddings,
)
from src.model.modeling.ahus_model import AhusModel


def decoder_forward(
    model: AhusModel,
    image_embeddings: list[BatchedImageEmbedding],
    mask_logits: BatchedImageLogits | None = None,
    points: tuple[BatchedPointCoords, BatchedPointLabels] | None = None,
    boxes: BatchedBox | None = None,  # or BatchedMulticlassBoxes?
) -> tuple[BatchedImageLogits, BatchedPromptEmbeddings]:
    batch_size, _, *spatial_dims = image_embeddings[0].shape
    sparse_emb, sparse_emb_pe_term, sparse_emb_pe_factor, dense_emb = model.prompt_encoder(
        spatial_dims, points, boxes, mask_logits
    )
    mask_logits, encoded_prompts = model.mask_decoder(
        image_embeddings,
        model.prompt_encoder.get_dense_pe_term(batch_size, spatial_dims),
        model.prompt_encoder.get_dense_pe_factor(batch_size, spatial_dims),
        sparse_emb,
        sparse_emb_pe_term,
        sparse_emb_pe_factor,
        dense_emb,
    )
    return mask_logits, encoded_prompts
