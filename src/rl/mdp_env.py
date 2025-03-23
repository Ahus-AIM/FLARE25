from typing import Iterator

import torch
from tensordict import TensorDict, TensorDictBase  # type: ignore
from torchrl.data import Binary, Bounded, Composite, Unbounded  # type: ignore
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs import EnvBase  # type: ignore

from src.custom_types import (  # highres_mask_shape,
    IMAGE_DEPTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ImageEmbedderFn,
    InteractionFn,
    MaskFn,
    MedicalData,
    PostProcessingFn,
    RewardFn,
    bbox_shape,
    done_shape,
    embedding_shape,
    image_shape,
    lowres_mask_shape,
    point_label_shape,
    point_shape,
    reward_shape,
    segmentation_shape,
    step_shape,
    threshold_shape,
)


class InteractiveSegmentationEnv(EnvBase):  # type: ignore
    # TODO: make env work without batched_locked
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
        batch_size: torch.Size,
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
            batch_size: batch size
        """
        super().__init__(device=device, batch_size=batch_size)  # type: ignore

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
        # TODO
        max_coord = max(IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.observation_spec: TensorSpec = Composite(
            image=Bounded(
                low=0,
                high=1,
                shape=self.batch_size + image_shape,
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding=Unbounded(
                shape=self.batch_size + embedding_shape,
                dtype=torch.float32,
                domain="continuous",
            ),
            bbox=Bounded(
                low=0, high=max_coord, shape=self.batch_size + bbox_shape, dtype=torch.float32, domain="continuous"
            ),
            low_res_mask=Unbounded(
                shape=self.batch_size + lowres_mask_shape,
                dtype=torch.float32,
                domain="continuous",
            ),
            points=Bounded(
                low=0,
                high=max_coord,
                shape=self.batch_size + (self.n_steps,) + point_shape,
                dtype=torch.int64,
                domain="discrete",
            ),
            point_labels=Bounded(
                low=0,
                high=1,
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
            true_segmentation=Bounded(
                low=0, high=1, shape=self.batch_size + segmentation_shape, dtype=torch.int64, domain="discrete"
            ),
            shape=self.batch_size,
        )
        self.action_spec: TensorSpec = Composite(
            # sampling_method=Categorical(3, shape=(), dtype=torch.int64),
            threshold=Bounded(
                low=0, high=1, shape=self.batch_size + threshold_shape, dtype=torch.float32, domain="continuous"
            ),
            shape=self.batch_size,
        )
        self.reward_spec: TensorSpec = Unbounded(
            shape=self.batch_size + reward_shape, dtype=torch.float32, domain="continuous"
        )
        self.done_spec: TensorSpec = Binary(shape=self.batch_size + done_shape, dtype=torch.bool)

    def _reset(self, _) -> TensorDict:
        # if tensordict is None:
        #     batch_size = torch.Size()
        # else:
        #     batch_size = tensordict.shape

        data = next(self.dataset_iter)
        image, true_segmentation, bbox = data["image"], data["label"], data["boxes"]

        # Move all data to the device
        image = image.to(self.device)
        true_segmentation = true_segmentation.to(self.device)
        bbox = bbox.to(self.device)

        # Data batch dimension should be the same as env batch dimension
        assert torch.Size((image.size(0),)) == self.batch_size

        image_embedding = self.image_embedder_fn(image)

        return TensorDict(
            {
                "image": image,
                "image_embedding": image_embedding,
                "bbox": bbox,
                "low_res_mask": torch.zeros(
                    self.batch_size + lowres_mask_shape, dtype=torch.float32, device=self.device
                ),
                "points": torch.zeros(
                    self.batch_size + (self.n_steps,) + point_shape, dtype=torch.int64, device=self.device
                ),
                "point_labels": torch.zeros(
                    self.batch_size + (self.n_steps,) + point_label_shape, dtype=torch.int64, device=self.device
                ),
                "step": torch.zeros(self.batch_size + step_shape, dtype=torch.int64, device=self.device),
                "true_segmentation": true_segmentation,
                "done": torch.full(self.batch_size + done_shape, False, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None):
        pass

    def _step(self, tensordict: TensorDictBase):
        # Only the first "steps" points and labels have meaningful values
        # NOTE: we assume that "step" is the same for each batch item
        step: int = tensordict["step"][0]
        new_low_res_mask = self.mask_fn(
            tensordict["image_embedding"],
            tensordict["bbox"],
            tensordict["points"][..., :step, :],
            tensordict["point_labels"][..., :step],
            tensordict["low_res_mask"],
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

        done = torch.full(self.batch_size + done_shape, False, dtype=torch.bool, device=self.device)
        done[tensordict["step"] + 1 == self.n_steps] = True

        return TensorDict(
            {
                "image": tensordict["image"],
                "image_embedding": tensordict["image_embedding"],
                "bbox": tensordict["bbox"],
                "low_res_mask": new_low_res_mask,
                "points": new_points,
                "point_labels": new_point_labels,
                "step": tensordict["step"] + 1,
                "true_segmentation": tensordict["true_segmentation"],
                "done": done,
                "reward": reward,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
