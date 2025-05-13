from typing import Callable, Iterator

import torch
from jaxtyping import Float
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs import EnvBase

from src.custom_types import (
    DONE_SHAPE,
    POINT_COORD_SHAPE,
    POINT_LABEL_SHAPE,
    REWARD_SHAPE,
    SINGLE_BOX_SHAPE,
    STEP_SHAPE,
    BatchedImageLogits,
    BatchedPointCoord,
    BatchedPointCoords,
    BatchedPointLabel,
    BatchedPointLabels,
    BatchedPromptEmbeddings,
    BatchedSegmentation,
    Boxes,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    ImageLogitsFn,
    InteractionFn,
    MulticlassSegmentation,
    PointCoord,
    PostProcessingFn,
    Reward,
    RewardFn,
    Step,
)
from src.model.modeling.ahus_model import AhusModel
from src.rl.utils import (
    image_logits_to_multiclass_segmentation,
    multiclass_to_singleclass_segmentations,
    pad_prompt_embeddings,
)
from src.transform.transform import VolumeTransforms
from src.utils.decode import decoder_forward
from src.utils.interact import interact
from src.utils.reward import compute_multi_class_dsc_nsd


class InteractiveSegmentationEnv(EnvBase):
    batched_locked = True

    def __init__(
        self,
        n_steps: int,
        image_embedder_fn: ImageEmbedderFn,
        image_logits_fn: ImageLogitsFn,
        post_processing_fn: PostProcessingFn,
        interaction_fn: InteractionFn,
        # downsample_fn: Callable,
        # upsample_fn: Callable,
        reward_fn: RewardFn,
        td_iter: Iterator[TensorDict],
        device: torch.device,
        size_threshold: int,
    ):
        """
        Args:
            n_steps: number of steps in the MDP
            image_embedder_fn: function that embeds an image
            mask_fn: function that generates a new low resolution mask
            post_processing_fn: function that generates a new high resolution segmentation
            # downsample_fn: function that can downsample both images and coordinates
            # upsample_fn: function that can upsample both images and coordinates
            reward_fn: function that computes the reward
            dataset_iter: iterator over the dataset with batch size 1
            device: device to use
            size_threshold: size threshold for downsampling
        """
        super().__init__(device=device, batch_size=torch.Size((1,)))  # only supported for batch size 1

        self.batch_size: torch.Size  # set by __init__
        self.device: torch.device  # set by __init__

        self.n_steps: int = n_steps
        self.image_embedder_fn: ImageEmbedderFn = image_embedder_fn
        self.image_logits_fn: ImageLogitsFn = image_logits_fn
        self.post_processing_fn: PostProcessingFn = post_processing_fn
        self.interaction_fn: InteractionFn = interaction_fn
        # self.downsample_fn: Callable = downsample_fn
        # self.upsample_fn: Callable = upsample_fn
        self.reward_fn: RewardFn = reward_fn
        self.td_iter: Iterator[TensorDict] = td_iter
        self.size_threshold: int = size_threshold

        self._make_spec()

    def _make_spec(self):
        self.observation_spec: TensorSpec = Composite(
            # Image has a single channel with values between 0 and 1
            downsampled_image=Bounded(
                low=0,
                high=1,
                shape=self.batch_size + (-1, -1, -1),
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
            padded_prompt_embeddings=Unbounded(
                shape=self.batch_size + (-1, self.n_steps + 2, -1),  # bbox + number of points(steps)
                dtype=torch.float32,
                domain="continuous",
            ),
            # Since prompt_embeddings is padded, we need a mask to ignore the padding
            prompt_embedding_attention_mask=Binary(
                shape=self.batch_size
                + (
                    -1,
                    self.n_steps + 2,
                ),
                dtype=torch.bool,
                device=self.device,
            ),
            # One bbox per instance
            downsampled_boxes=Unbounded(
                shape=self.batch_size + (-1,) + SINGLE_BOX_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            downsampled_image_logits=Unbounded(
                shape=self.batch_size + (-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            # point_coords and point_labels are actually bounded, but we don't know the image size beforehand
            downsampled_point_coords=Unbounded(
                shape=self.batch_size
                + (
                    -1,
                    self.n_steps,
                )
                + POINT_COORD_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            point_labels=Unbounded(
                shape=self.batch_size
                + (
                    -1,
                    self.n_steps,
                )
                + POINT_LABEL_SHAPE,
                dtype=torch.int64,
                domain="discrete",
            ),
            step=Bounded(
                low=0,
                high=self.n_steps - 1,
                shape=self.batch_size + STEP_SHAPE,
                dtype=torch.int64,
                domain="discrete",
            ),
            true_multiclass_segmentation=Unbounded(
                shape=self.batch_size + (-1, -1, -1),
                dtype=torch.uint8,
                domain="discrete",
            ),
            spacing=Unbounded(
                shape=self.batch_size + (-1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            shape=self.batch_size,
        )
        # We have as many actions as there are instances
        self.action_spec: TensorSpec = Composite(
            # sampling_method=Categorical(3, shape=(), dtype=torch.int64),
            logits_to_add=Unbounded(
                shape=self.batch_size + (-1, 1),
                dtype=torch.float32,
                domain="continuous",
            ),
            shape=self.batch_size,
        )
        self.reward_spec: TensorSpec = Unbounded(
            shape=self.batch_size + REWARD_SHAPE,
            dtype=torch.float32,
            domain="continuous",
        )
        self.done_spec: TensorSpec = Binary(shape=self.batch_size + DONE_SHAPE, dtype=torch.bool)

    def _reset(self, tensordict, **kwargs) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict({}, device=self.device, batch_size=self.batch_size)
        data_td = next(self.td_iter)
        image, true_multiclass_segmentation, boxes, spacing = (
            data_td["image"],
            data_td["true_multiclass_segmentation"],
            data_td["boxes"],
            data_td["spacing"],
        )
        # The number of instances is determined by the label
        n_instances = int(data_td["true_multiclass_segmentation"].unique().numel() - 1)

        # Move all data to the device
        image = image.to(tensordict.device)
        true_multiclass_segmentation = true_multiclass_segmentation.to(tensordict.device)
        boxes = boxes.to(tensordict.device)
        spacing = spacing.to(tensordict.device)

        # Downscale everything that will be passed to the image logits model
        self.coord_handler = VolumeTransforms(size_threshold=self.size_threshold)
        downsampled_image, downsampled_boxes, _ = self.coord_handler.forward(volume=image, boxes=boxes)

        image_embeddings = self.image_embedder_fn(downsampled_image)

        # Produce initial image logits and prompt embeddings
        downsampled_image_logits, prompt_embeddings = self.image_logits_fn(
            [
                image_embeddings[0],
                image_embeddings[1],
                image_embeddings[2],
                image_embeddings[3],
            ],
            downsampled_boxes if downsampled_boxes.shape[0] > 0 else None,
            None,
            None,
            None,
        )

        # Pad prompt embeddings
        padded_prompt_embeddings, prompt_embedding_attention_mask = pad_prompt_embeddings(
            prompt_embeddings, self.n_steps
        )

        td = TensorDict(
            {
                "downsampled_image": downsampled_image,
                "image_embedding1": image_embeddings[0],
                "image_embedding2": image_embeddings[1],
                "image_embedding3": image_embeddings[2],
                "image_embedding4": image_embeddings[3],
                "padded_prompt_embeddings": padded_prompt_embeddings,
                "prompt_embedding_attention_mask": prompt_embedding_attention_mask,
                "downsampled_boxes": downsampled_boxes,
                "downsampled_image_logits": downsampled_image_logits,
                "downsampled_point_coords": torch.zeros(
                    (
                        n_instances,
                        self.n_steps,
                    )
                    + POINT_COORD_SHAPE,
                    dtype=torch.float32,
                    device=tensordict.device,
                ),
                "point_labels": torch.zeros(
                    (
                        n_instances,
                        self.n_steps,
                    )
                    + POINT_LABEL_SHAPE,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "step": torch.zeros(
                    STEP_SHAPE,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "true_multiclass_segmentation": true_multiclass_segmentation,
                "spacing": spacing,
                "done": torch.full(
                    DONE_SHAPE,
                    False,
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
            },
            device=tensordict.device,
        )
        return td.unsqueeze(0)  # batch size 1

    def _set_seed(self, seed: int | None):
        pass

    def _step(self, tensordict: TensorDictBase):
        # Only batch size 1 is supported
        assert tensordict.shape == (1,)
        tensordict = tensordict.squeeze(0)

        # From now on, dimension 1 is the class instance dimension

        n_instances = tensordict["downsampled_image_logits"].shape[0]

        # Only the first "step" points and labels have meaningful values
        step: int = tensordict["step"][0]

        # Multiclass segmentation, influenced by agent's "logits_to_add"
        # Needed for reward
        image_logits = self.coord_handler.backward(tensordict["downsampled_image_logits"])
        multiclass_segmentation = image_logits_to_multiclass_segmentation(
            image_logits + tensordict["logits_to_add"].view(n_instances, 1, 1, 1),
        )

        # Reward depends on multiclass segmentation and potentially on the step
        reward = self.reward_fn(
            multiclass_segmentation,
            tensordict["true_multiclass_segmentation"],
            tensordict["step"],
            tensordict["spacing"],
        )

        # Interaction occurs in upsampled space
        new_point_coord, new_point_label = self.interaction_fn(
            multiclass_segmentation,
            tensordict["true_multiclass_segmentation"],
            n_instances,
        )

        # Downsample the points so they can be passed to the image logits model
        downsampled_new_point_coord_batched = self.coord_handler.pool_coords(new_point_coord.unsqueeze(0))
        downsampled_new_point_coord: PointCoord = downsampled_new_point_coord_batched.squeeze(0)

        # Add the points to the tensors
        downsampled_new_point_coords = tensordict["downsampled_point_coords"].clone()
        downsampled_new_point_coords[:, [step], :] = downsampled_new_point_coord
        new_point_labels = tensordict["point_labels"].clone()
        new_point_labels[:, [step]] = new_point_label

        # Update the mask and prompt embeddings using the new points
        downsampled_new_image_logits, new_prompt_embeddings = self.image_logits_fn(
            [
                tensordict["image_embedding1"],
                tensordict["image_embedding2"],
                tensordict["image_embedding3"],
                tensordict["image_embedding4"],
            ],
            tensordict["downsampled_boxes"] if tensordict["downsampled_boxes"].shape[0] > 0 else None,
            downsampled_new_point_coords[:, : step + 1],
            new_point_labels[:, : step + 1],
            tensordict["downsampled_image_logits"],
        )

        # Pad prompt embeddings
        padded_prompt_embeddings, prompt_embedding_attention_mask = pad_prompt_embeddings(
            new_prompt_embeddings, self.n_steps
        )

        done = torch.full(
            DONE_SHAPE,
            False,
            dtype=torch.bool,
            device=tensordict.device,
        )
        done[tensordict["step"] + 1 == self.n_steps] = True

        td = TensorDict(
            {
                "downsampled_image": tensordict["downsampled_image"],
                "image_embedding1": tensordict["image_embedding1"],
                "image_embedding2": tensordict["image_embedding2"],
                "image_embedding3": tensordict["image_embedding3"],
                "image_embedding4": tensordict["image_embedding4"],
                "padded_prompt_embeddings": padded_prompt_embeddings,
                "prompt_embedding_attention_mask": prompt_embedding_attention_mask,
                "downsampled_boxes": tensordict["downsampled_boxes"],
                "downsampled_image_logits": downsampled_new_image_logits,
                "downsampled_point_coords": downsampled_new_point_coords,
                "point_labels": new_point_labels,
                "step": tensordict["step"] + 1,
                "true_multiclass_segmentation": tensordict["true_multiclass_segmentation"],
                "spacing": tensordict["spacing"],
                "done": done,
                "reward": reward,
            },
            device=tensordict.device,
        )

        return td.unsqueeze(0)  # batch size 1


def get_image_embedder_fn(
    ahus_model: AhusModel,
    ahus_model_device: torch.device,
    env_device: torch.device,
    image_transform_fn: Callable[[Image], Image] | None = None,
) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> list[ImageEmbedding]:
        # Optionally apply a transform to the image, most commonly downsampling
        if image_transform_fn is not None:
            image = image_transform_fn(image)
        # Ahus model expects a batch and channel dimension
        image_embeddings = ahus_model.image_encoder(image.to(ahus_model_device).unsqueeze(0).unsqueeze(0))
        # Move the embeddings to the env device and remove batch dimension
        image_embeddings = [emb.to(env_device).squeeze(0) for emb in image_embeddings]
        return image_embeddings

    return image_embedder_fn


def get_image_logits_fn(
    ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device
) -> ImageLogitsFn:
    def mask_fn(
        image_embeddings: list[ImageEmbedding],
        boxes: Boxes | None,
        point_coords: BatchedPointCoords | None,
        point_labels: BatchedPointLabels | None,
        image_logits: BatchedImageLogits | None,
    ) -> tuple[BatchedImageLogits, BatchedPromptEmbeddings]:
        # decoder_forward expects everything in batched format.
        # It also expects
        # - one image embedding per instance
        # - image logits should have a channel dimension
        n_instances = boxes.shape[0] if boxes is not None else 1

        # decoder_forward will run out of GPU memory if we forward everything in one go
        # Therefore, we only forward one instance at a time
        new_image_logits_list = []
        new_prompt_embeddings_list = []

        # Same image embeddings for all instances
        image_embeddings = [emb.to(ahus_model_device).unsqueeze(0) for emb in reversed(image_embeddings)]

        for i in range(n_instances):
            instance_box = boxes[i : i + 1].to(ahus_model_device) if boxes is not None else None
            instance_point_coords = point_coords[i : i + 1].to(ahus_model_device) if point_coords is not None else None
            instance_point_labels = point_labels[i : i + 1].to(ahus_model_device) if point_labels is not None else None
            instance_image_logits = (
                image_logits[i : i + 1].to(ahus_model_device).unsqueeze(1) if image_logits is not None else None
            )

            # Assume that if either point_coords or point_labels is None, then both are None
            if instance_point_coords is None or instance_point_labels is None:
                instance_points = None
            else:
                instance_points = (instance_point_coords, instance_point_labels)

            new_instance_image_logits, new_instance_prompt_embeddings = decoder_forward(
                ahus_model,
                image_embeddings,
                instance_image_logits,
                instance_points,
                instance_box,
            )

            # Move the mask logits and prompt embeddings to the env device
            new_image_logits_list.append(new_instance_image_logits.squeeze(0).squeeze(0).to(env_device))
            new_prompt_embeddings_list.append(new_instance_prompt_embeddings.squeeze(0).to(env_device))

        new_image_logits: BatchedImageLogits = torch.stack(new_image_logits_list, dim=0)
        new_prompt_embeddings: BatchedPromptEmbeddings = torch.stack(new_prompt_embeddings_list, dim=0)

        return new_image_logits, new_prompt_embeddings

    return mask_fn


def get_post_processing_fn() -> PostProcessingFn:
    def post_processing_fn(
        image: Image,
        image_logits: BatchedImageLogits,
        logits_to_add: Float[Tensor, "batch 1"],
    ) -> BatchedSegmentation:
        summed_logits = image_logits + logits_to_add.view(logits_to_add.shape[0], 1, 1, 1)
        return (summed_logits > 0.0).to()

    return post_processing_fn


def get_interaction_fn(interact_device: torch.device) -> InteractionFn:
    def interaction_fn(
        multiclass_segmentation: MulticlassSegmentation,
        true_multiclass_segmentation: MulticlassSegmentation,
        n_instances: int,
    ) -> tuple[BatchedPointCoord, BatchedPointLabel]:
        # The interact function expects singleclass segmentations
        singleclass_segmentations = multiclass_to_singleclass_segmentations(
            multiclass_segmentation,
            n_instances,
        )
        true_singleclass_segmentations = multiclass_to_singleclass_segmentations(
            true_multiclass_segmentation,
            n_instances,
        )
        point_coord_list, point_label_list = interact(
            singleclass_segmentations.unsqueeze(1).long().to(interact_device),
            true_singleclass_segmentations.unsqueeze(1).long().to(interact_device),
        )
        # Assume that we only receive one point coord and label per batch
        point_coord: BatchedPointCoord = torch.cat(point_coord_list, dim=0).to(multiclass_segmentation.device)
        point_label: BatchedPointLabel = torch.cat(point_label_list, dim=0).to(multiclass_segmentation.device)
        return point_coord, point_label

    return interaction_fn


def get_reward_fn() -> RewardFn:
    # TODO: use step
    def reward_fn(
        seg: MulticlassSegmentation,
        true_seg: MulticlassSegmentation,
        step: Step,
        spacing: torch.Tensor,
    ) -> Reward:
        dsc_tensor, nsd_tensor = compute_multi_class_dsc_nsd(
            true_seg,
            seg,
            spacing=spacing,
        )
        return (dsc_tensor + nsd_tensor).unsqueeze(-1)  # normalization

    return reward_fn


def infinite_loader(factory: Callable[[], Iterator]):
    while True:
        iterator = factory()
        for elem in iterator:
            yield elem


# def get_downsample_fn(TODO) -> Callable:
#     """"""
#     def downsample_fn(*args) -> tuple[Tensor, ...]:


def get_env(
    ahus_model: AhusModel,
    ahus_model_device: torch.device,
    env_device: torch.device,
    td_iterator_factory: Callable[[], Iterator[TensorDict]],
    size_threshold: int,
    interact_device: torch.device,
):
    env = InteractiveSegmentationEnv(
        n_steps=5,
        image_embedder_fn=get_image_embedder_fn(ahus_model, ahus_model_device, env_device=env_device),
        image_logits_fn=get_image_logits_fn(ahus_model, ahus_model_device=ahus_model_device, env_device=env_device),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(interact_device),
        reward_fn=get_reward_fn(),
        td_iter=infinite_loader(
            td_iterator_factory,
        ),
        device=env_device,
        size_threshold=size_threshold,
    )
    return env
