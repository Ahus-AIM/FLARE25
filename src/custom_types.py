from collections.abc import Callable

import torch
from beartype.typing import List, Tuple, TypedDict
from jaxtyping import Float, Integer
from torch import Tensor

BBOX_SHAPE = torch.Size((2, 3))
POINT_COORD_SHAPE = torch.Size((3,))
POINT_LABEL_SHAPE = torch.Size(())
THRESHOLD_SHAPE = torch.Size((1,))
STEP_SHAPE = torch.Size((1,))
REWARD_SHAPE = torch.Size((1,))
DONE_SHAPE = torch.Size((1,))
UPSAMPLING_METHOD_SHAPE = torch.Size(())


def size_to_str(s: torch.Size) -> str:
    return " ".join(str(x) for x in s)


Image = Float[Tensor, "batch image_channel image_depth image_height image_width"]
Mask = Float[Tensor, "batch image_channel image_depth image_height image_width"]
Segmentation = Integer[Tensor, "batch image_channel image_depth image_height image_width"]
# Image embeddings can have varied number of channels and spatial dimensions
ImageEmbedding = Float[Tensor, "batch _embedding_channel _embedding_depth _embedding_height _embedding_width"]

BBox = Float[Tensor, f"batch {size_to_str(BBOX_SHAPE)}"]
PointCoord = Float[Tensor, f"batch {size_to_str(POINT_COORD_SHAPE)}"]
PointCoords = Float[Tensor, f"batch n_points {size_to_str(POINT_COORD_SHAPE)}"]
PointLabel = Integer[Tensor, f"batch {size_to_str(POINT_LABEL_SHAPE)}"]
PointLabels = Integer[Tensor, f"batch n_points {size_to_str(POINT_LABEL_SHAPE)}"]
Threshold = Float[Tensor, f"batch {size_to_str(THRESHOLD_SHAPE)}"]
UpsamplingMethod = Integer[Tensor, f"batch {size_to_str(UPSAMPLING_METHOD_SHAPE)}"]
Step = Integer[Tensor, f"batch {size_to_str(STEP_SHAPE)}"]
Reward = Float[Tensor, f"batch {size_to_str(REWARD_SHAPE)}"]

# An image embedder function returns a list of image embeddings, due to the unet nature of the model
ImageEmbedderFn = Callable[[Image], List[ImageEmbedding]]

MaskFn = Callable[
    [List[ImageEmbedding], BBox, PointCoords, PointLabels, Mask],  # previous mask
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
    Tuple[PointCoord, PointLabel],  # new point and label
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
