from collections.abc import Callable

import torch
from beartype.typing import List, Tuple, TypedDict
from jaxtyping import Float, Integer
from torch import Tensor

bbox_shape = torch.Size((2, 3))
point_shape = torch.Size((3,))
point_label_shape = torch.Size(())
threshold_shape = torch.Size(())
step_shape = torch.Size(())
reward_shape = torch.Size((1,))
done_shape = torch.Size((1,))
upsampling_method_shape = torch.Size(())


def size_to_str(s: torch.Size) -> str:
    return " ".join(str(x) for x in s)


Image = Float[Tensor, "batch image_channel image_depth image_height image_width"]
Mask = Float[Tensor, "batch image_channel image_depth image_height image_width"]
Segmentation = Integer[Tensor, "batch image_channel image_depth image_height image_width"]
# Image embeddings can have varied number of channels and spatial dimensions
ImageEmbedding = Float[Tensor, "batch _embedding_channel _embedding_depth _embedding_height _embedding_width"]

BBox = Float[Tensor, f"batch {size_to_str(bbox_shape)}"]
Point = Integer[Tensor, f"batch {size_to_str(point_shape)}"]
Points = Integer[Tensor, f"batch n_points {size_to_str(point_shape)}"]
PointLabel = Integer[Tensor, f"batch {size_to_str(point_label_shape)}"]
PointLabels = Integer[Tensor, f"batch n_points {size_to_str(point_label_shape)}"]
Threshold = Float[Tensor, f"batch {size_to_str(threshold_shape)}"]
UpsamplingMethod = Integer[Tensor, f"batch {size_to_str(upsampling_method_shape)}"]
Step = Integer[Tensor, f"batch {size_to_str(step_shape)}"]
Reward = Float[Tensor, f"batch {size_to_str(reward_shape)}"]

# An image embedder function returns a list of image embeddings, due to the unet nature of the model
ImageEmbedderFn = Callable[[Image], List[ImageEmbedding]]

MaskFn = Callable[
    [List[ImageEmbedding], BBox, Points, PointLabels, Mask],  # previous mask
    Mask,  # new mask
]

PostProcessingFn = Callable[
    [
        Image,
        Mask,
        # UpsamplingMethod,
        Threshold,
    ],
    Segmentation,  # model segmentation
]

InteractionFn = Callable[
    [
        Segmentation,  # model segmentation
        Segmentation,  # true segmentation
    ],
    Tuple[Point, PointLabel],  # new point and label
]

RewardFn = Callable[
    [
        Segmentation,  # model segmentation
        Segmentation,  # true segmentation
        Step,
    ],
    Reward,
]


class MedicalData(TypedDict):
    image: Image
    boxes: BBox
    label: Segmentation
