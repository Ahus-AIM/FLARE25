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
MulticlassImageLogits = Float[Tensor, "n_instances image_depth image_height image_width"]
Segmentation = Bool[Tensor, "image_depth image_height image_width"]
MulticlassSegmentation = Integer[Tensor, "image_depth image_height image_width"]
ImageEmbedding = Float[Tensor, "_embedding_channel _embedding_depth _embedding_height _embedding_width"]
PromptEmbeddings = Float[Tensor, "n_points prompt_embedding_size"]
Box = Float[Tensor, f"{size_to_str(SINGLE_BOX_SHAPE)}"]
PointCoord = Float[Tensor, f"{size_to_str(POINT_COORD_SHAPE)}"]
PointCoords = Float[Tensor, f"n_points {size_to_str(POINT_COORD_SHAPE)}"]
MulticlassPointCoords = Float[Tensor, f"n_instances n_points {size_to_str(POINT_COORD_SHAPE)}"]
PointLabel = Integer[Tensor, f"{size_to_str(POINT_LABEL_SHAPE)}"]
PointLabels = Integer[Tensor, f"n_points {size_to_str(POINT_LABEL_SHAPE)}"]
MulticlassPointLabels = Integer[Tensor, f"n_instances n_points {size_to_str(POINT_LABEL_SHAPE)}"]
Step = Integer[Tensor, f"{size_to_str(STEP_SHAPE)}"]
Reward = Float[Tensor, f"{size_to_str(REWARD_SHAPE)}"]
Boxes = Float[Tensor, f"n_instances {size_to_str(SINGLE_BOX_SHAPE)}"]

# An image embedder function returns a list of image embeddings, due to the unet nature of the model
ImageEmbedderFn = Callable[[Image], list[ImageEmbedding]]

# A logits mask function is responsible for
# 1. Producing a mask of logits.
# 2. Producing new prompt embeddings.
# All inputs except the image embeddings are optional, since image segmentation should be possible without labels.
ImageLogitsFn = Callable[
    [list[ImageEmbedding], Box | None, PointCoords | None, PointLabels | None, ImageLogits | None],  # previous mask
    tuple[ImageLogits, PromptEmbeddings],  # new mask
]

# Postprocessing applies to multiple instances
PostProcessingFn = Callable[
    [
        Image,
        MulticlassImageLogits,  # model logits
        MulticlassImageLogits,  # action: add logits
    ],
    MulticlassSegmentation,
]

# Interaction applies to a single instance
InteractionFn = Callable[
    [
        Segmentation,  # predicted segmentation
        Segmentation,  # true segmentation
    ],
    tuple[PointCoord, PointLabel],  # new point and label
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


# Batched versions
BatchedImage = Float[Tensor, "batch image_depth image_height image_width"]
BatchedImageLogits = Float[Tensor, "batch image_depth image_height image_width"]
BatchedSegmentation = Integer[Tensor, "batch image_depth image_height image_width"]
BatchedImageEmbedding = Float[Tensor, "batch _embedding_channel _embedding_depth _embedding_height _embedding_width"]
BatchedPromptEmbeddings = Float[Tensor, "batch n_points prompt_embedding_size"]
BatchedBox = Float[Tensor, f"batch {size_to_str(SINGLE_BOX_SHAPE)}"]
BatchedPointCoord = Float[Tensor, f"batch {size_to_str(POINT_COORD_SHAPE)}"]
BatchedPointCoords = Float[Tensor, f"batch n_points {size_to_str(POINT_COORD_SHAPE)}"]
BatchedPointLabel = Integer[Tensor, f"batch {size_to_str(POINT_LABEL_SHAPE)}"]
BatchedPointLabels = Integer[Tensor, f"batch n_points {size_to_str(POINT_LABEL_SHAPE)}"]
BatchedMulticlassBoxes = Float[Tensor, f"batch n_instances {size_to_str(SINGLE_BOX_SHAPE)}"]

BatchedChannelImage = Float[Tensor, "batch 1 image_depth image_height image_width"]
