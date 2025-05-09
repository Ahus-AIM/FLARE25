from types import SimpleNamespace

import torch
from torch import nn

from src.custom_types import BatchedPromptAttentionMask, BatchedPromptEmbeddings


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
        (batch_size, n_steps + 2, prompt_embedding_size),  # bbox + number of points(steps)
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
