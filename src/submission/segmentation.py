from typing import Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Integer
from tensordict import TensorDict

from src.custom_types import BBox, Image, Mask, PointCoords, PointLabels, Segmentation, Threshold
from src.rl.agents import PPOThresholdAgent


class Segmenter(Protocol):
    """
    A segmenter is responsible for converting a mask of logits into a binary segmentation.
    """

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


class PPOThresholdAgentSegmenter(Segmenter):
    """
    Wraps a PPOThresholdAgent to implement the Segmenter protocol.
    """

    def __init__(self, agent: PPOThresholdAgent):
        self.agent = agent

    def __call__(
        self, logits: Mask, image: Image, bbox: BBox, point_coords: PointCoords | None, point_labels: PointLabels | None
    ) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        # TODO
        # Create a tensordict with the correct format
        td = TensorDict(
            {
                "mask": logits,
            },
            batch_size=logits.shape[:1],
            device=logits.device,
        )
        # Pass tensordict to agent, creating a "threshold" key
        td = self.agent.policy(td)
        threshold: Threshold = td["threshold"]
        # Apply the threshold to the logits to get the segmentation
        segmentation: Segmentation = F.sigmoid(logits) > threshold
        return segmentation

    @staticmethod
    def load(path: str, device: torch.device) -> "Segmenter":
        agent = PPOThresholdAgent.load(path, device)
        return PPOThresholdAgentSegmenter(agent)


class OriginalSegmenter(Segmenter):
    """
    Constant threshold segmenter that also ensures that all classes are present in the segmentation.
    """

    def __init__(self):
        pass

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
        return OriginalSegmenter()


segmenter_registry = {
    "ppo_threshold": PPOThresholdAgentSegmenter,
    "original": OriginalSegmenter,
}
