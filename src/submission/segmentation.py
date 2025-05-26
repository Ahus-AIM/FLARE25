from pathlib import Path
from typing import Callable, Protocol, Self

import torch
from jaxtyping import Integer
from tensordict import TensorDict
from torch import Tensor

from src.custom_types import BatchedImageLogits, Box, Boxes, ImageLogits, PointCoords
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent
from src.rl.mdp_env import MulticlassSegmentation
from src.rl.utils import image_logits_to_multiclass_segmentation


class Segmenter(Protocol):
    """
    A segmenter is responsible for producing a multiclass binary segmentation.
    """

    device: torch.device

    def __call__(
        self,
        image_logits: BatchedImageLogits | None,
        downsampled_image_logits: BatchedImageLogits | None,
        boxes: Boxes | None,
        downsampled_boxes: Boxes | None,
        point_coords: PointCoords | None,
        downsampled_point_coords: PointCoords | None,
        point_labels: BatchedImageLogits | None,
        padded_prompt_embeddings: Tensor | None,
        prompt_embedding_attension_mask: Tensor | None,
    ) -> MulticlassSegmentation:
        """
        Performs segmentation on a single image, with multiple bounding boxes and points per instance.

        Outputs: an integer tensor where each pixel is assigned a class. 0 represents the background, and 1 to n represent the classes.
        """
        ...

    @classmethod
    def load(cls, path: Path | None, device: torch.device) -> Self:
        """
        Loads the segmenter from a path and places it on the device.
        """
        ...


def add_to_logits(logits: ImageLogits, box: Box, box_margin: int = 1, increment=1.0) -> None:
    """
    Adds a value to the logits in the bounding box defined by box_i. Modifies the logits in place.
    The box coordinates are inclusive, so we add 1 to the end coordinates but also ensure that they are within the bounds of the logits.

    logits: (D, H, W)
    """
    D, H, W = logits.shape
    box = box.clone().round().int()
    z0, y0, x0 = box[0].tolist()
    z1, y1, x1 = torch.minimum(box[1] + 1, torch.tensor([D, H, W], device=box.device)).tolist()

    logits[z0:z1, y0:y1, x0:x1] += increment


def thresholded_argmax_segmentation(
    td: TensorDict,
    segment_fn: Callable[[TensorDict], None],
) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
    """
    Produces a multi-class segmentation map from thresholded instance predictions.

    Args:
        td: TensorDict of shape (N,) with the following keys:
            - "mask": logits of shape (N, 1, D, H, W)
        segment_fn: function that writes to the "multiclass_segmentation" key of the TensorDict, shape (N, D, H, W).

    Returns:
        seg: LongTensor of shape (D, H, W), with values in [0, N].
             0 means no confident instance predicted that pixel.
             i means instance i gave the strongest confident prediction at that pixel.
    """
    N, C, D, H, W = td["mask"].shape
    assert C == 1, "Expected logits to have a channel dimension of 1"
    # print(f"Logits shape: {logits.shape}")
    # device = logits.device

    # Writes "segmentation" key to the TensorDict
    segment_fn(td)
    # Mask out logits where the prediction is not confident
    valid_logits = torch.where(td["segmentation"], td["mask"], float("-inf"))  # (N, D, H, W)

    # Add a dummy background logit (class 0)
    background = torch.zeros(1, 1, D, H, W, device=td.device)
    padded_logits = torch.cat([background, valid_logits], dim=0)

    # Argmax gives class index in [0, N]
    return padded_logits.argmax(dim=0).squeeze(0)  # (D, H, W)


class AttentionPPOThresholdAgentSegmenter(Segmenter):
    """
    Wraps AttentionPPOThresholdAgent to implement the Segmenter protocol.
    """

    def __init__(self, agent: AttentionPPOThresholdAgent, max_iter: int = 10):
        self.agent = agent
        self.device = self.agent.device
        self.max_iter = max_iter

    @classmethod
    def load(cls, path: Path | None, device: torch.device) -> Self:
        assert path is not None, "Path must be provided to load the segmenter"
        agent = AttentionPPOThresholdAgent.load(path)
        agent.device = device
        return cls(agent)

    def __call__(
        self,
        image_logits: BatchedImageLogits | None,
        downsampled_image_logits: BatchedImageLogits | None,
        boxes: Boxes | None,
        downsampled_boxes: Boxes | None,
        point_coords: PointCoords | None,
        downsampled_point_coords: PointCoords | None,
        point_labels: BatchedImageLogits | None,
        padded_prompt_embeddings: Tensor | None,
        prompt_embedding_attension_mask: Tensor | None,
    ) -> MulticlassSegmentation:
        assert (
            padded_prompt_embeddings is not None
        ), "Padded prompt embeddings must be provided to AttentionPPOThresholdAgentSegmenter"
        assert (
            prompt_embedding_attension_mask is not None
        ), "Prompt embedding attention mask must be provided to AttentionPPOThresholdAgentSegmenter"

        n_instances = padded_prompt_embeddings.shape[0]
        # Create a tensordict in the format expected by the agent
        td = TensorDict(
            {
                "padded_prompt_embeddings": padded_prompt_embeddings,
                "prompt_embedding_attention_mask": prompt_embedding_attension_mask,
            },
            batch_size=(),
            device=self.device,
        )
        # Agent expects a batch dimension
        td = td.unsqueeze(0)
        td = self.agent.policy(td)
        td = td.squeeze(0)
        return image_logits_to_multiclass_segmentation(image_logits + td["logits_to_add"].view(n_instances, 1, 1, 1))


class DummySegmenter(Segmenter):
    """Constant threshold segmenter that does not modify the logits."""

    def __init__(self, device: torch.device):
        self.device = device
        self.threshold = 0.0

    def __call__(
        self,
        image_logits: BatchedImageLogits | None,
        downsampled_image_logits: BatchedImageLogits | None,
        boxes: Boxes | None,
        downsampled_boxes: Boxes | None,
        point_coords: PointCoords | None,
        downsampled_point_coords: PointCoords | None,
        point_labels: BatchedImageLogits | None,
        padded_prompt_embeddings: Tensor | None,
        prompt_embedding_attension_mask: Tensor | None,
    ) -> MulticlassSegmentation:
        assert image_logits is not None, "Image logits must be provided to DummySegmenter"

        return image_logits_to_multiclass_segmentation(image_logits + self.threshold)

    @classmethod
    def load(cls, path: Path | None, device: torch.device) -> Self:
        return cls(device)


class OriginalSegmenter(Segmenter):
    """
    Constant threshold segmenter that also ensures that all classes are present in the segmentation.
    """

    def __init__(self, device: torch.device):
        self.device = device  # expected by Segmenter protocol

    def __call__(
        self,
        image_logits: BatchedImageLogits | None,
        downsampled_image_logits: BatchedImageLogits | None,
        boxes: Boxes | None,
        downsampled_boxes: Boxes | None,
        point_coords: PointCoords | None,
        downsampled_point_coords: PointCoords | None,
        point_labels: BatchedImageLogits | None,
        padded_prompt_embeddings: Tensor | None,
        prompt_embedding_attension_mask: Tensor | None,
    ) -> MulticlassSegmentation:
        assert image_logits is not None, "Image logits must be provided to OriginalSegmenter"
        if boxes is None:
            boxes = torch.zeros((image_logits.shape[0], 2, 3), device=image_logits.device)

        # ensure everything is on the CPU
        image_logits = image_logits.cpu()
        boxes = boxes.cpu()

        max_iter = 3
        ensure_all_present = True
        # threshold_value = 0.5
        # print("image_logits dtype:", image_logits.dtype)
        # print("threshold_value type:", type(threshold_value))
        # print("autocast is enabled:", torch.is_autocast_enabled())
        threshold_tensor = torch.full_like(image_logits[0:1], 0.0)
        n_instances = image_logits.shape[0]

        not_all_instances_present = True
        counter = 0
        while not_all_instances_present and counter < max_iter:
            pred_concat = torch.cat((threshold_tensor, image_logits), dim=0)
            pred_long = pred_concat.argmax(dim=0)

            if not ensure_all_present:
                return pred_long

            # For instances that are not yet present, we add a value to the logits in their bounding box
            present_instances = torch.unique(pred_long)
            for i, cls in enumerate(range(1, n_instances + 1)):  # for each instance
                if cls not in present_instances:
                    add_to_logits(image_logits[i], boxes[i], increment=2**counter)

            counter += 1

            not_all_instances_present = len(torch.unique(pred_long)) != n_instances + 1

        # Is not unbounded
        return pred_long  # type: ignore

    @classmethod
    def load(cls, path: Path | None, device: torch.device) -> Self:
        # Nothing to load
        return cls(device)


segmenter_registry = {
    "attention_ppo_threshold": AttentionPPOThresholdAgentSegmenter,
    "original": OriginalSegmenter,
    "dummy": DummySegmenter,
}
