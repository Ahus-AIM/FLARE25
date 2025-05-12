from types import SimpleNamespace

import torch
from torch import nn

from src.custom_types import (
    BatchedImageLogits,
    BatchedPromptAttentionMask,
    BatchedPromptEmbeddings,
    BatchedSegmentation,
    MulticlassSegmentation,
)


def calculate_norm(module: nn.Module) -> float:
    # Aggregate all parameters from the module and compute the norm
    return torch.norm(torch.cat([p.view(-1) for p in module.parameters()]), p=2).item()  # L2 norm (Euclidean norm)


def dict_to_namespace(d):
    """Convert a dict to a SimpleNamespace recursively."""
    if not isinstance(d, dict):
        return d

    # Create a namespace for this level
    ns = SimpleNamespace()

    # Convert each key-value pair
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries
            setattr(ns, key, dict_to_namespace(value))
        elif isinstance(value, list):
            # Convert lists with potential nested dictionaries
            setattr(
                ns,
                key,
                [dict_to_namespace(item) if isinstance(item, dict) else item for item in value],
            )
        else:
            # Set the attribute directly for primitive types
            setattr(ns, key, value)

    return ns


def pad_prompt_embeddings(
    prompt_embeddings: BatchedPromptEmbeddings, n_steps: int
) -> tuple[BatchedPromptEmbeddings, BatchedPromptAttentionMask]:
    """

    Args:
        prompt_embeddings: Float tensor of shape (batch_size, s, prompt_embedding_size) where s depends on the current number of points received.
        n_steps: Number of feedback steps used to train the RL agent.
    Returns:
        padded_prompt_embeddings: Float tensor of shape (batch_size, n_steps+2, prompt_embedding_size)
        prompt_embeddings_mask: Binary tensor of shape (batch_size, n_steps+2,)

    """

    prompt_embedding_size = prompt_embeddings.shape[-1]
    batch_size = prompt_embeddings.shape[0]
    device = prompt_embeddings.device

    padded_prompt_embeddings = torch.zeros(
        (
            batch_size,
            n_steps + 2,
            prompt_embedding_size,
        ),  # bbox + number of points(steps)
        dtype=torch.float32,
        device=device,
    )
    padded_prompt_embeddings[:, : prompt_embeddings.shape[1], :] = prompt_embeddings
    prompt_embedding_attention_mask = torch.full(
        (
            batch_size,
            n_steps + 2,
        ),
        False,
        dtype=torch.bool,
        device=device,
    )
    prompt_embedding_attention_mask[:, : prompt_embeddings.shape[1]] = True

    return padded_prompt_embeddings, prompt_embedding_attention_mask


def image_logits_to_multiclass_segmentation(
    image_logits: BatchedImageLogits,
) -> MulticlassSegmentation:
    # Logits that are below the background threshold (0) are set to 0
    image_logits[image_logits < 0.0] = 0
    background_and_logits = torch.cat(
        [
            torch.ones_like(image_logits[0:1]),  # we can use anything here as long as it's larger than 0
            image_logits,
        ],
        dim=0,
    )

    return background_and_logits.argmax(dim=0).to(torch.uint8)


def singleclass_segmentations_to_multiclass_segmentation(
    singleclass_segmentations: BatchedSegmentation,
) -> MulticlassSegmentation:
    background_and_singleclass_segmentations = torch.cat(
        [
            2
            * torch.ones_like(singleclass_segmentations[0:1]),  # we can use anything here as long as it's larger than 1
            singleclass_segmentations,
        ],
        dim=0,
    )

    return background_and_singleclass_segmentations.argmax(dim=0).to(torch.uint8)


def multiclass_to_singleclass_segmentations(
    multiclass_segmentation: MulticlassSegmentation,
    n_classes: int,
) -> BatchedSegmentation:
    singleclass_segmentations = torch.zeros((n_classes, *multiclass_segmentation.shape), dtype=torch.bool).to(
        multiclass_segmentation.device
    )

    for i, cls in enumerate(range(1, n_classes + 1)):
        singleclass_segmentations[i] = multiclass_segmentation == cls

    return singleclass_segmentations
