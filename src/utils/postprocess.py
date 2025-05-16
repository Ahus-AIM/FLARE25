# from src.custom_types import ImageLogits, MulticlassSegmentation, Threshold


# def standard_threshold(mask_logits: ImageLogits, threshold: Threshold) -> MulticlassSegmentation:
#     # Expand threshold to have same shape as mask_logits
#     threshold = threshold.view(-1, 1, 1, 1, 1).expand_as(mask_logits)
#     return (mask_logits.sigmoid() > threshold).long()
