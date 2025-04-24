from types import SimpleNamespace

import torch
from torch import nn


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
