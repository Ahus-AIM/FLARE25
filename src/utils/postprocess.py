from beartype import beartype
from jaxtyping import jaxtyped

from custom_types import Mask, Segmentation, Threshold


@jaxtyped(typechecker=beartype)
def standard_threshold(mask_logits: Mask, threshold: Threshold) -> Segmentation:
    # Expand threshold to have same shape as mask_logits
    threshold = threshold.view(-1, 1, 1, 1, 1).expand_as(mask_logits)
    return (mask_logits.sigmoid() > threshold).long()
