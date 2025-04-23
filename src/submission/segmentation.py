from typing import Callable, Protocol

import torch
from jaxtyping import Integer
from tensordict import TensorDict

from src.custom_types import BBox, Image, Mask, PointCoords, PointLabels, Segmentation
from src.rl.agents import DDPGThresholdAgent


class Segmenter(Protocol):
    """
    A segmenter is responsible for converting a mask of logits into a binary segmentation.
    """

    device: torch.device

    def __call__(
        self, logits: Mask, image: Image, bbox: BBox, point_coords: PointCoords | None, point_labels: PointLabels | None
    ) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        """
        Performs segmentation on a single image, with multiple bounding boxes and points. The batch dimension in bbox, point_coords and point_labels represents different classes.

        Outputs: an integer tensor where each pixel is assigned a class. 0 represents the background, and 1 to n represent the classes.
        """
        ...

    @staticmethod
    def load(path: str, device: torch.device) -> "Segmenter":
        """
        Loads the segmenter from a path and places it on the device.
        """
        ...


def thresholded_argmax_segmentation(
    logits: Mask,
    image: Image,
    bboxes: BBox,
    point_coords: PointCoords | None,
    point_labels: PointLabels | None,
    segment_fn: Callable[
        [Mask, Image, BBox, PointCoords | None, PointLabels | None], Segmentation
    ],  # returns (N, 1, D, H, W) bool
) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
    """
    Produces a multi-class segmentation map from thresholded instance predictions.

    Returns:
        seg: LongTensor of shape (D, H, W), with values in [0, N].
             0 means no confident instance predicted that pixel.
             i means instance i gave the strongest confident prediction at that pixel.
    """
    N, C, D, H, W = logits.shape
    # print(f"Logits shape: {logits.shape}")
    device = logits.device

    # Mask out logits where the prediction is not confident
    confident_mask = segment_fn(logits, image, bboxes, point_coords, point_labels)  # (N, D, H, W)
    valid_logits = torch.where(confident_mask, logits, float("-inf"))  # (N, D, H, W)

    # Add a dummy background logit (class 0)
    background = torch.zeros(1, 1, D, H, W, device=device)
    padded_logits = torch.cat([background, valid_logits], dim=0)

    # Argmax gives class index in [0, N]
    return padded_logits.argmax(dim=0).squeeze(0)  # (D, H, W)


class DDPGThresholdAgentSegmenter(Segmenter):
    """
    Wraps a PPOThresholdAgent to implement the Segmenter protocol.
    """

    def __init__(self, agent: DDPGThresholdAgent, max_iter: int = 10):
        self.agent = agent
        self.device = self.agent.device
        self.max_iter = max_iter

    def __call__(
        self,
        logits: Mask,
        image: Image,
        bbox: BBox,
        point_coords: PointCoords | None,
        point_labels: PointLabels | None,
    ) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        counter = 0
        all_instances_present = True

        while all_instances_present and counter < self.max_iter:
            # Get threshold from PPO agent
            def segment_fn(
                logits: Mask,
                image: Image,
                bboxes: BBox,
                point_coords: PointCoords | None,
                point_labels: PointLabels | None,
            ) -> Mask:
                # Get the threshold from the PPO agent
                td = TensorDict({"mask": logits}, batch_size=logits.shape[:1], device=logits.device)
                td = self.agent.policy_module(td)
                threshold = td["threshold"]
                return logits > threshold.reshape(-1, 1, 1, 1, 1)

            pred_long = thresholded_argmax_segmentation(
                logits,
                image,
                bbox,
                point_coords,
                point_labels,
                segment_fn=segment_fn,
            )

            # Check class presence
            present_instances = torch.unique(pred_long)
            if len(present_instances) == logits.shape[0] + 1:
                break

            # Add to logits for missing instances
            for i in range(logits.shape[0]):
                if (i + 1) not in present_instances:
                    logits[i] = self._add_to_logits(logits[i], bbox[i], increment=2**counter)

            counter += 1
            all_instances_present = True

        return pred_long  # type: ignore

    def _add_to_logits(
        self, logits: torch.Tensor, box_i: torch.Tensor, box_margin: int = 1, increment: float = 1.0
    ) -> torch.Tensor:
        """
        Adds a value to the logits in the bounding box defined by box_i.
        """
        box = box_i.clone().round().int()
        box[0] = torch.clamp(box[0] - box_margin, 0, logits.shape[-1])
        box[1] = torch.clamp(box[1] + box_margin, 0, logits.shape[-1])
        logits[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]] += increment
        return logits

    @staticmethod
    def load(path: str, device: torch.device) -> "Segmenter":
        agent = DDPGThresholdAgent.load(path, device)
        return DDPGThresholdAgentSegmenter(agent)


class OriginalSegmenter(Segmenter):
    """
    Constant threshold segmenter that also ensures that all classes are present in the segmentation.
    """

    def __init__(self, device: torch.device):
        self.device = device  # expected by Segmenter protocol

    def __call__(
        self, logits: Mask, image: Image, bbox: BBox, point_coords: PointCoords | None, point_labels: PointLabels | None
    ) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        max_iter = 10
        ensure_all_present = True
        threshold = 0.5

        all_instances_present = True
        counter = 0
        while all_instances_present and counter < max_iter:
            pred_prob = torch.sigmoid(logits)
            pred_concat = torch.cat((torch.ones_like(logits)[0:1] * threshold, pred_prob), dim=0)
            pred_long = pred_concat.argmax(dim=0).squeeze(0)

            if not ensure_all_present:
                return pred_long

            # For instances that are not yet present, we add a value to the logits in their bounding box
            present_instances = torch.unique(pred_long)
            for i in range(logits.shape[0]):  # for each instance
                if (i + 1) not in present_instances:
                    logits[i] = self._add_to_logits(logits[i], bbox[i], increment=2**counter)

            counter += 1

            all_instances_present = len(torch.unique(pred_long)) != logits.shape[0] + 1

        # Is not unbounded
        return pred_long  # type: ignore

    def _add_to_logits(
        self, logits: torch.Tensor, box_i: torch.Tensor, box_margin: int = 1, increment=1.0
    ) -> torch.Tensor:
        """
        Adds a value to the logits in the bounding box defined by box_i.
        """
        box = box_i.clone().round().int()
        box[0] = torch.clamp(box[0] - box_margin, 0, logits.shape[-1])
        box[1] = torch.clamp(box[1] + box_margin, 0, logits.shape[-1])
        logits[0, box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]] += increment
        return logits

    @staticmethod
    def load(path: str, device: torch.device) -> "Segmenter":
        # Nothing to load
        return OriginalSegmenter(device)


segmenter_registry = {
    "ddpg_threshold": DDPGThresholdAgentSegmenter,
    "original": OriginalSegmenter,
}
