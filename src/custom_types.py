from collections.abc import Callable
from typing import Tuple, TypedDict

import torch
from jaxtyping import Float, Integer
from torch import Tensor

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 128
LOWRES_MASK_WIDTH = 32
LOWRES_MASK_HEIGHT = 32
LOWRES_MASK_DEPTH = 32
EMBEDDING_DIM = 64
EMBEDDING_CHANNELS = 384
EMBEDDING_DEPTH = 8
EMBEDDING_WIDTH = 8
EMBEDDING_HEIGHT = 8
HIGHRES_MASK_DEPTH = 128
HIGHRES_MASK_WIDTH = 128
HIGHRES_MASK_HEIGHT = 128

lowres_mask_shape = torch.Size((1, LOWRES_MASK_DEPTH, LOWRES_MASK_HEIGHT, LOWRES_MASK_WIDTH))
highres_mask_shape = torch.Size((1, HIGHRES_MASK_DEPTH, HIGHRES_MASK_HEIGHT, HIGHRES_MASK_WIDTH))
image_shape = torch.Size((1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH))
embedding_shape = torch.Size((EMBEDDING_CHANNELS, EMBEDDING_DEPTH, EMBEDDING_HEIGHT, EMBEDDING_WIDTH))
bbox_shape = torch.Size((2, 3))
point_shape = torch.Size((3,))
segmentation_shape = torch.Size((1, HIGHRES_MASK_DEPTH, HIGHRES_MASK_HEIGHT, HIGHRES_MASK_WIDTH))
point_label_shape = torch.Size(())
threshold_shape = torch.Size(())
step_shape = torch.Size(())
reward_shape = torch.Size((1,))
done_shape = torch.Size((1,))
upsampling_method_shape = torch.Size(())


def size_to_str(s: torch.Size) -> str:
    return " ".join(str(x) for x in s)


Image = Float[Tensor, f"batch {size_to_str(image_shape)}"]
LowresMask = Float[Tensor, f"batch {size_to_str(lowres_mask_shape)}"]
ImageEmbedding = Float[Tensor, f"batch {size_to_str(embedding_shape)}"]
HighresMask = Float[Tensor, f"batch {size_to_str(highres_mask_shape)}"]
BBox = Float[Tensor, f"batch {size_to_str(bbox_shape)}"]
Point = Integer[Tensor, f"batch {size_to_str(point_shape)}"]
Points = Integer[Tensor, f"batch n_points {size_to_str(point_shape)}"]
PointLabel = Integer[Tensor, f"batch {size_to_str(point_label_shape)}"]
PointLabels = Integer[Tensor, f"batch n_points {size_to_str(point_label_shape)}"]
Segmentation = Integer[Tensor, f"batch {size_to_str(segmentation_shape)}"]
Threshold = Float[Tensor, f"batch {size_to_str(threshold_shape)}"]
UpsamplingMethod = Integer[Tensor, f"batch {size_to_str(upsampling_method_shape)}"]
Step = Integer[Tensor, f"batch {size_to_str(step_shape)}"]
Reward = Float[Tensor, f"batch {size_to_str(reward_shape)}"]

ImageEmbedderFn = Callable[[Image], ImageEmbedding]

MaskFn = Callable[
    [
        ImageEmbedding,
        BBox,
        Points,
        PointLabels,
        LowresMask,  # previous low resolution mask
    ],
    LowresMask,  # new low resolution mask
]

PostProcessingFn = Callable[
    [
        Image,
        LowresMask,
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
