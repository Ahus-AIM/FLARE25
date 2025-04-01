from typing import Iterator

import torch
from tensordict import TensorDict, TensorDictBase  # type: ignore
from torchrl.data import Binary, Bounded, Composite, Unbounded  # type: ignore
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs import EnvBase  # type: ignore

from src.custom_types import (
    ImageEmbedderFn,
    InteractionFn,
    MaskFn,
    MedicalData,
    PostProcessingFn,
    RewardFn,
    bbox_shape,
    done_shape,
    point_label_shape,
    point_shape,
    reward_shape,
    step_shape,
    threshold_shape,
)


class InteractiveSegmentationEnv(EnvBase):
    batched_locked = True

    def __init__(
        self,
        n_steps: int,
        image_embedder_fn: ImageEmbedderFn,
        mask_fn: MaskFn,
        post_processing_fn: PostProcessingFn,
        interaction_fn: InteractionFn,
        reward_fn: RewardFn,
        dataset_iter: Iterator[MedicalData],
        device: torch.device,
    ):
        """
        Args:
            n_steps: number of steps in the MDP
            image_embedder_fn: function that embeds an image
            mask_fn: function that generates a new low resolution mask
            post_processing_fn: function that generates a new high resolution segmentation
            reward_fn: function that computes the reward
            dataset_iter: iterator over the dataset
            device: device to use
        """
        super().__init__(device=device, batch_size=torch.Size((1,)))  # always use batch size of 1

        self.batch_size: torch.Size  # set by __init__
        self.device: torch.device  # set by __init__

        self.n_steps: int = n_steps
        self.image_embedder_fn: ImageEmbedderFn = image_embedder_fn
        self.mask_fn: MaskFn = mask_fn
        self.post_processing_fn: PostProcessingFn = post_processing_fn
        self.interaction_fn: InteractionFn = interaction_fn
        self.reward_fn: RewardFn = reward_fn
        self.dataset_iter: Iterator[MedicalData] = dataset_iter

        self._make_spec()

    def _make_spec(self):
        self.observation_spec: TensorSpec = Composite(
            # image has three channels (RGB)
            image=Bounded(
                low=0,
                high=1,
                shape=self.batch_size + (1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding1=Unbounded(
                shape=self.batch_size + (-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding2=Unbounded(
                shape=self.batch_size + (-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding3=Unbounded(
                shape=self.batch_size + (-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding4=Unbounded(
                shape=self.batch_size + (-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            # bbox is actually bounded, but we don't know the image size beforehand
            bbox=Unbounded(
                shape=self.batch_size + bbox_shape,
                dtype=torch.float32,
                domain="continuous",
            ),
            # mask has a single channel, same shape as image
            mask=Unbounded(
                shape=self.batch_size + (1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            # points and point_labels are actually bounded, but we don't know the image size beforehand
            points=Unbounded(
                shape=self.batch_size + (self.n_steps,) + point_shape,
                dtype=torch.int64,
                domain="discrete",
            ),
            point_labels=Unbounded(
                shape=self.batch_size + (self.n_steps,) + point_label_shape,
                dtype=torch.int64,
                domain="discrete",
            ),
            step=Bounded(
                low=0,
                high=self.n_steps - 1,
                shape=self.batch_size + step_shape,
                dtype=torch.int64,
                domain="discrete",
            ),
            # segmentation has same shape as image
            true_segmentation=Bounded(
                low=0,
                high=1,
                shape=self.batch_size + (1, -1, -1, -1),
                dtype=torch.int64,
                domain="discrete",
            ),
            shape=self.batch_size,
        )
        self.action_spec: TensorSpec = Composite(
            # sampling_method=Categorical(3, shape=(), dtype=torch.int64),
            threshold=Bounded(
                low=0,
                high=1,
                shape=self.batch_size + threshold_shape,
                dtype=torch.float32,
                domain="continuous",
            ),
            shape=self.batch_size,
        )
        self.reward_spec: TensorSpec = Unbounded(
            shape=self.batch_size + reward_shape,
            dtype=torch.float32,
            domain="continuous",
        )
        self.done_spec: TensorSpec = Binary(shape=self.batch_size + done_shape, dtype=torch.bool)

    def _reset(self, tensordict, **kwargs) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        data = next(self.dataset_iter)
        image, true_segmentation, bbox = data["image"], data["label"], data["boxes"]

        # Move all data to the device
        image = image.to(tensordict.device)
        true_segmentation = true_segmentation.to(tensordict.device)
        bbox = bbox.to(tensordict.device)

        assert (
            torch.Size((image.size(0),)) == tensordict.batch_size
        ), "Data batch dimension should be the same as env batch dimension"

        image_embeddings = self.image_embedder_fn(image)

        td = TensorDict(
            {
                "image": image,
                "image_embedding1": image_embeddings[0],
                "image_embedding2": image_embeddings[1],
                "image_embedding3": image_embeddings[2],
                "image_embedding4": image_embeddings[3],
                "bbox": bbox,
                "mask": torch.zeros_like(
                    image,
                    dtype=torch.float32,
                    device=tensordict.device,
                ),
                "points": torch.zeros(
                    tensordict.batch_size + (self.n_steps,) + point_shape,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "point_labels": torch.zeros(
                    tensordict.batch_size + (self.n_steps,) + point_label_shape,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "step": torch.zeros(tensordict.batch_size + step_shape, dtype=torch.int64, device=tensordict.device),
                "true_segmentation": true_segmentation,
                "done": torch.full(
                    tensordict.batch_size + done_shape,
                    False,
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
        return td

    def _set_seed(self, seed: int | None):
        pass

    def _step(self, tensordict: TensorDictBase):
        # Only the first "steps" points and labels have meaningful values
        step: int = tensordict["step"][0]
        new_low_res_mask = self.mask_fn(
            [
                tensordict["image_embedding1"],
                tensordict["image_embedding2"],
                tensordict["image_embedding3"],
                tensordict["image_embedding4"],
            ],
            tensordict["bbox"],
            tensordict["points"][..., :step, :],
            tensordict["point_labels"][..., :step],
            tensordict["mask"],
        )

        # Always use upsampling method 0 for now
        segmentation = self.post_processing_fn(tensordict["image"], new_low_res_mask, tensordict["threshold"])

        new_point, new_point_label = self.interaction_fn(segmentation, tensordict["true_segmentation"])
        # add new point and new label to the points and point_labels tensors
        new_points = tensordict["points"].clone()
        new_points[:, [step], :] = new_point
        new_point_labels = tensordict["point_labels"].clone()
        new_point_labels[:, [step]] = new_point_label

        reward = self.reward_fn(segmentation, tensordict["true_segmentation"], tensordict["step"])

        done = torch.full(tensordict.batch_size + done_shape, False, dtype=torch.bool, device=self.device)
        done[tensordict["step"] + 1 == self.n_steps] = True

        return TensorDict(
            {
                "image": tensordict["image"],
                "image_embedding1": tensordict["image_embedding1"],
                "image_embedding2": tensordict["image_embedding2"],
                "image_embedding3": tensordict["image_embedding3"],
                "image_embedding4": tensordict["image_embedding4"],
                "bbox": tensordict["bbox"],
                "mask": new_low_res_mask,
                "points": new_points,
                "point_labels": new_point_labels,
                "step": tensordict["step"] + 1,
                "true_segmentation": tensordict["true_segmentation"],
                "done": done,
                "reward": reward,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
