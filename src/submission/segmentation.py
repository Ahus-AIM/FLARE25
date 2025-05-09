from pathlib import Path
from typing import Callable, Protocol, Self

import torch
from jaxtyping import Integer
from tensordict import TensorDict

from src.rl.agents import Agent


class Segmenter(Protocol):
    """
    A segmenter is responsible for producing a multiclass binary segmentation.
    """

    device: torch.device

    def __call__(self, td: TensorDict) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        """
        Performs segmentation on a single image, with multiple bounding boxes and points. The batch dimension in bbox, point_coords and point_labels represents different classes. Since each segmenter may require different inputs, it takes a TensorDict as input.

        Outputs: an integer tensor where each pixel is assigned a class. 0 represents the background, and 1 to n represent the classes.
        """
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Loads the segmenter from a path and places it on the device.
        """
        ...


def add_to_logits(logits: torch.Tensor, box_i: torch.Tensor, box_margin: int = 1, increment=1.0) -> torch.Tensor:
    """
    Adds a value to the logits in the bounding box defined by box_i.

    logits: (D, H, W)
    box_i: (2, 3) tensor with the coordinates of the bounding box
    """
    box = box_i.clone().round().int()
    logits[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]] += increment
    return logits


def thresholded_argmax_segmentation(
    td: TensorDict,
    segment_fn: Callable[[TensorDict], None],
) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
    """
    Produces a multi-class segmentation map from thresholded instance predictions.

    Args:
        td: TensorDict of shape (N,) with the following keys:
            - "mask": logits of shape (N, 1, D, H, W)
        segment_fn: function that writes to the "segmentation" key of the TensorDict, shape (N, 1, D, H, W).

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


class AgentSegmenter(Segmenter):
    """
    Wraps an Agent to implement the Segmenter protocol.
    """

    def __init__(self, agent: Agent, max_iter: int = 10):
        self.agent = agent
        self.device = self.agent.device
        self.max_iter = max_iter

    def __call__(self, td: TensorDict) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        """
        Args:
            td: TensorDict of shape (N,) with the following keys:
                - "padded_prompt_embeddings"
                - "prompt_embedding_attention_mask
        """
        counter = 0
        all_instances_present = True

        while all_instances_present and counter < self.max_iter:
            # Get threshold from PPO agent
            def segment_fn(td: TensorDict) -> None:
                self.agent.policy(td)
                td["segmentation"] = td["mask"] > td["threshold"].reshape(-1, 1, 1, 1, 1)

            pred_long = thresholded_argmax_segmentation(
                td,
                segment_fn=segment_fn,
            )

            # Check class presence
            present_instances = torch.unique(pred_long)
            if len(present_instances) == td["mask"].shape[0] + 1:
                break

            # Add to logits for missing instances
            for i in range(td["mask"].shape[0]):
                if (i + 1) not in present_instances:
                    add_to_logits(td["mask"][i, 0], td["bbox"][i], increment=2**counter)

            counter += 1
            all_instances_present = True

        return pred_long  # type: ignore

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        agent = Agent.load(path)
        agent.device = device
        return cls(agent)


class OriginalSegmenter(Segmenter):
    """
    Constant threshold segmenter that also ensures that all classes are present in the segmentation.
    """

    def __init__(self, device: torch.device):
        self.device = device  # expected by Segmenter protocol

    def __call__(self, td: TensorDict) -> Integer[torch.Tensor, "image_depth image_height image_width"]:
        """
        Args:
            td: TensorDict of shape (N,) with the following keys:
                - "mask": logits of shape (N, 1, D, H, W)
                - "bbox": bounding boxes of shape (N, 2, 3)
        """
        max_iter = 10
        ensure_all_present = True
        threshold = 0.5

        not_all_instances_present = True
        counter = 0
        while not_all_instances_present and counter < max_iter:
            pred_prob = torch.sigmoid(td["mask"])
            pred_concat = torch.cat((torch.ones_like(td["mask"])[0:1] * threshold, pred_prob), dim=0)
            pred_long = pred_concat.argmax(dim=0).squeeze(0)

            if not ensure_all_present:
                return pred_long

            # For instances that are not yet present, we add a value to the logits in their bounding box
            present_instances = torch.unique(pred_long)
            for i in range(td["mask"].shape[0]):  # for each instance
                if (i + 1) not in present_instances:
                    add_to_logits(td["mask"][i, 0], td["bbox"][i], increment=2**counter)

            counter += 1

            not_all_instances_present = len(torch.unique(pred_long)) != td["mask"].shape[0] + 1

        # Is not unbounded
        return pred_long  # type: ignore

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        # Nothing to load
        return cls(device)


segmenter_registry = {
    # "ddpg_threshold": DDPGThresholdAgentSegmenter,
    "agent_segmenter": AgentSegmenter,
    "original": OriginalSegmenter,
}
