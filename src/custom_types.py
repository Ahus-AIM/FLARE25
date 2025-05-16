from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

SINGLE_BOX_SHAPE = torch.Size((2, 3))
POINT_COORD_SHAPE = torch.Size((3,))
POINT_LABEL_SHAPE = torch.Size(())
THRESHOLD_SHAPE = torch.Size((1,))
STEP_SHAPE = torch.Size((1,))
REWARD_SHAPE = torch.Size((1,))
DONE_SHAPE = torch.Size((1,))
UPSAMPLING_METHOD_SHAPE = torch.Size(())


def size_to_str(s: torch.Size) -> str:
    return " ".join(str(x) for x in s)


Image = Float[Tensor, "image_depth image_height image_width"]
ImageLogits = Float[Tensor, "image_depth image_height image_width"]
Segmentation = Bool[Tensor, "image_depth image_height image_width"]
ChannelSegmentation = Bool[Tensor, "1 image_depth image_height image_width"]
MulticlassSegmentation = Integer[Tensor, "image_depth image_height image_width"]
ImageEmbedding = Float[Tensor, "_embedding_channel _embedding_depth _embedding_height _embedding_width"]
PromptEmbeddings = Float[Tensor, "n_points prompt_embedding_size"]
PromptAttentionMask = Bool[Tensor, "n_points"]
Box = Float[Tensor, f"{size_to_str(SINGLE_BOX_SHAPE)}"]
PointCoord = Float[Tensor, f"{size_to_str(POINT_COORD_SHAPE)}"]
PointCoords = Float[Tensor, f"n_points {size_to_str(POINT_COORD_SHAPE)}"]
PointLabel = Integer[Tensor, f"{size_to_str(POINT_LABEL_SHAPE)}"]
PointLabels = Integer[Tensor, f"n_points {size_to_str(POINT_LABEL_SHAPE)}"]
Step = Integer[Tensor, f"{size_to_str(STEP_SHAPE)}"]
Reward = Float[Tensor, f"{size_to_str(REWARD_SHAPE)}"]
Boxes = Float[Tensor, f"n_instances {size_to_str(SINGLE_BOX_SHAPE)}"]

# Batched versions
BatchedImage = Float[Tensor, "batch image_depth image_height image_width"]
BatchedImageLogits = Float[Tensor, "batch image_depth image_height image_width"]
BatchedChannelImageLogits = Float[Tensor, "batch 1 image_depth image_height image_width"]
BatchedSegmentation = Bool[Tensor, "batch image_depth image_height image_width"]
BatchedChannelSegmentation = Bool[Tensor, "batch 1 image_depth image_height image_width"]
BatchedImageEmbedding = Float[Tensor, "batch _embedding_channel _embedding_depth _embedding_height _embedding_width"]
BatchedPromptEmbeddings = Float[Tensor, "batch n_points prompt_embedding_size"]
BatchedPromptAttentionMask = Bool[Tensor, "batch n_points"]
BatchedBox = Float[Tensor, f"batch {size_to_str(SINGLE_BOX_SHAPE)}"]
BatchedPointCoord = Float[Tensor, f"batch {size_to_str(POINT_COORD_SHAPE)}"]
BatchedPointCoords = Float[Tensor, f"batch n_points {size_to_str(POINT_COORD_SHAPE)}"]
BatchedPointLabel = Integer[Tensor, f"batch {size_to_str(POINT_LABEL_SHAPE)}"]
BatchedPointLabels = Integer[Tensor, f"batch n_points {size_to_str(POINT_LABEL_SHAPE)}"]

BatchedChannelImage = Float[Tensor, "batch 1 image_depth image_height image_width"]

# An image embedder function returns a list of image embeddings, due to the unet nature of the model
ImageEmbedderFn = Callable[[Image], list[ImageEmbedding]]

# A logits mask function is responsible for
# 1. Producing a mask of logits.
# 2. Producing new prompt embeddings.
# All inputs except the image embeddings are optional, since image segmentation should be possible without labels.
ImageLogitsFn = Callable[
    [
        list[ImageEmbedding],
        Boxes | None,
        BatchedPointCoords | None,
        BatchedPointLabels | None,
        BatchedImageLogits | None,
    ],  # previous mask
    tuple[BatchedImageLogits, BatchedPromptEmbeddings],  # new mask
]

# Postprocessing applies to multiple instances
PostProcessingFn = Callable[
    [
        Image,
        BatchedImageLogits,  # model logits
        BatchedImageLogits,  # action: add logits
    ],
    MulticlassSegmentation,
]

# Interaction applies to multiclass segmentation and returns
# one point per class.
InteractionFn = Callable[
    [
        MulticlassSegmentation,  # predicted segmentation
        MulticlassSegmentation,  # true segmentation
        int,  # number of instances
    ],
    tuple[BatchedPointCoord, BatchedPointLabel],  # new point and label per class
]

# Reward is calculated from multiple instances
RewardFn = Callable[
    [
        MulticlassSegmentation,  # predicted segmentation
        MulticlassSegmentation,  # true segmentation
        Step,
        Tensor,  # spacing
    ],
    Reward,
]


# class MedicalData(TypedDict):
#     image: Image
#     boxes: MulticlassBoxes
#     label: Segmentation
