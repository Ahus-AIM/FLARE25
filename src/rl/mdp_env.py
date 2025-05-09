from typing import Callable, Iterator

import torch
from tensordict import TensorDict, TensorDictBase
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
    Box,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    ImageLogits,
    ImageLogitsFn,
    InteractionFn,
    MulticlassImageLogits,
    MulticlassSegmentation,
    PointCoord,
    PointCoords,
    PointLabel,
    PointLabels,
    PostProcessingFn,
    PromptEmbeddings,
    Reward,
    RewardFn,
    Step,
)
from src.model.modeling import AhusModel
from src.rl.utils import pad_prompt_embeddings
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
        reward_fn: RewardFn,
        td_iter: Iterator[TensorDict],
        device: torch.device,
    ):
        """
        Args:
            n_steps: number of steps in the MDP
            image_embedder_fn: function that embeds an image
            mask_fn: function that generates a new low resolution mask
            post_processing_fn: function that generates a new high resolution segmentation
            reward_fn: function that computes the reward
            dataset_iter: iterator over the dataset with batch size 1
            device: device to use
        """
        super().__init__(
            device=device, batch_size=torch.Size(())
        )  # always use batch size of None

        # self.batch_size: torch.Size  # set by __init__
        self.device: torch.device  # set by __init__

        self.n_steps: int = n_steps
        self.image_embedder_fn: ImageEmbedderFn = image_embedder_fn
        self.image_logits_fn: ImageLogitsFn = image_logits_fn
        self.post_processing_fn: PostProcessingFn = post_processing_fn
        self.interaction_fn: InteractionFn = interaction_fn
        self.reward_fn: RewardFn = reward_fn
        self.td_iter: Iterator[TensorDict] = td_iter

        self._make_spec()

    def _make_spec(self):
        self.observation_spec: TensorSpec = Composite(
            # Image has a single channel with values between 0 and 1
            image=Bounded(
                low=0,
                high=1,
                shape=(-1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding1=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding2=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding3=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            image_embedding4=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            multiclass_padded_prompt_embeddings=Unbounded(
                shape=(-1, self.n_steps + 2, -1),  # bbox + number of points(steps)
                dtype=torch.float32,
                domain="continuous",
            ),
            # Since prompt_embeddings is padded, we need a mask to ignore the padding
            multiclass_prompt_embedding_attention_masks=Binary(
                shape=(
                    -1,
                    self.n_steps + 2,
                ),
                dtype=torch.bool,
                device=self.device,
            ),
            # One bbox per instance
            boxes=Unbounded(
                shape=(-1,) + SINGLE_BOX_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            # mask has a single channel, same shape as image
            multiclass_logits_masks=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            # point_coords and point_labels are actually bounded, but we don't know the image size beforehand
            multiclass_point_coords=Unbounded(
                shape=(
                    -1,
                    self.n_steps,
                )
                + POINT_COORD_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            multiclass_point_labels=Unbounded(
                shape=(
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
                shape=STEP_SHAPE,
                dtype=torch.int64,
                domain="discrete",
            ),
            true_multiclass_segmentation=Unbounded(
                shape=(-1, -1, -1),
                dtype=torch.uint8,
                domain="discrete",
            ),
        )
        # We have as many actions as there are instances
        self.action_spec: TensorSpec = Composite(
            # sampling_method=Categorical(3, shape=(), dtype=torch.int64),
            logits_to_add=Unbounded(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
        )
        self.reward_spec: TensorSpec = Unbounded(
            shape=REWARD_SHAPE,
            dtype=torch.float32,
            domain="continuous",
        )
        self.done_spec: TensorSpec = Binary(shape=DONE_SHAPE, dtype=torch.bool)

    def _reset(self, tensordict, **kwargs) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict({}, device=self.device)
        data_td = next(self.td_iter)
        image, true_multiclass_segmentation, boxes = (
            data_td["image"],
            data_td["true_multiclass_segmentation"],
            data_td["boxes"],
        )
        # The number of instances is how many boxes we have
        n_instances = boxes.shape

        # Move all data to the device
        image = image.to(tensordict.device)
        true_multiclass_segmentation = true_multiclass_segmentation.to(
            tensordict.device
        )
        boxes = boxes.to(tensordict.device)

        image_embeddings = self.image_embedder_fn(image)

        # Pass each instance through the mask function
        multiclass_image_logits = []
        multiclass_padded_prompt_embeddings = []
        multiclass_prompt_embedding_attention_masks = []
        for instance_idx in range(n_instances):
            # Produce initial mask and prompt embeddings
            image_logits, prompt_embeddings = self.image_logits_fn(
                [
                    image_embeddings[0],
                    image_embeddings[1],
                    image_embeddings[2],
                    image_embeddings[3],
                ],
                boxes[instance_idx],
                None,
                None,
                None,
            )
            padded_prompt_embeddings, prompt_embedding_attention_mask = (
                pad_prompt_embeddings(prompt_embeddings, self.n_steps)
            )
            multiclass_image_logits.append(image_logits)
            multiclass_padded_prompt_embeddings.append(padded_prompt_embeddings)
            multiclass_prompt_embedding_attention_masks.append(
                prompt_embedding_attention_mask
            )

        # Stack the results
        multiclass_image_logits = torch.stack(multiclass_image_logits, dim=0)
        multiclass_padded_prompt_embeddings = torch.stack(
            multiclass_padded_prompt_embeddings, dim=0
        )
        multiclass_prompt_embedding_attention_mask = torch.stack(
            multiclass_prompt_embedding_attention_masks, dim=0
        )

        td = TensorDict(
            {
                "image": image,
                "image_embedding1": image_embeddings[0],
                "image_embedding2": image_embeddings[1],
                "image_embedding3": image_embeddings[2],
                "image_embedding4": image_embeddings[3],
                "multiclass_padded_prompt_embeddings": multiclass_padded_prompt_embeddings,
                "multiclass_prompt_embedding_attention_masks": multiclass_prompt_embedding_attention_mask,
                "boxes": boxes,
                "multiclass_image_logits": multiclass_image_logits,
                "multiclass_point_coords": torch.zeros(
                    (
                        n_instances,
                        self.n_steps,
                    )
                    + POINT_COORD_SHAPE,
                    dtype=torch.float32,
                    device=tensordict.device,
                ),
                "multiclass_point_labels": torch.zeros(
                    (
                        n_instances,
                        self.n_steps,
                    )
                    + POINT_LABEL_SHAPE,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "step": torch.zeros(
                    tensordict.batch_size + STEP_SHAPE,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "true_multiclass_segmentation": true_multiclass_segmentation,
                "done": torch.full(
                    DONE_SHAPE,
                    False,
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
            },
            device=tensordict.device,
        )
        return td

    def _set_seed(self, seed: int | None):
        pass

    def _step(self, tensordict: TensorDictBase):
        # Only the first "steps" points and labels have meaningful values
        step: int = tensordict["step"][0]

        multiclass_segmentation = self.post_processing_fn(
            tensordict["image"],
            tensordict["multiclass_image_logits"],
            tensordict["logits_to_add"],
        )

        # Reward depends on segmentation
        reward = self.reward_fn(
            multiclass_segmentation, tensordict["true_segmentation"], tensordict["step"]
        )

        # Get new points for each instance, preallocate tensors
        n_instances = tensordict["multiclass_boxes"].shape[0]
        multiclass_padded_prompt_embeddings = tensordict[
            "multiclass_padded_prompt_embeddings"
        ].clone()
        multiclass_prompt_embedding_attention_masks = tensordict[
            "multiclass_prompt_embedding_attention_masks"
        ].clone()
        multiclass_image_logits = tensordict["multiclass_image_logits"].clone()
        multiclass_point_coords = tensordict["multiclass_point_coords"].clone()
        multiclass_point_labels = tensordict["multiclass_point_labels"].clone()
        for instance_idx in range(n_instances):
            singleclass_segmentation = multiclass_segmentation == instance_idx
            true_singleclass_segmentation = (
                tensordict["true_multiclass_segmentation"] == instance_idx
            )
            new_point_coord, new_point_label = self.interaction_fn(
                singleclass_segmentation, true_singleclass_segmentation
            )
            # add new point coord and new label to the point_coords and point_labels tensors
            new_point_coords = tensordict["multiclass_point_coords"][
                instance_idx
            ].clone()
            new_point_coords[:, [step], :] = new_point_coord
            new_point_labels = tensordict["multiclass_point_labels"][
                instance_idx
            ].clone()
            new_point_labels[:, [step]] = new_point_label

            # Update the mask and prompt embeddings using the new points
            new_image_logits, new_prompt_embeddings = self.image_logits_fn(
                [
                    tensordict["image_embedding1"],
                    tensordict["image_embedding2"],
                    tensordict["image_embedding3"],
                    tensordict["image_embedding4"],
                ],
                tensordict["boxes"][instance_idx],
                new_point_coords[: step + 1],
                new_point_labels[: step + 1],
                tensordict["multiclass_image_logits"][instance_idx],
            )
            # Pad prompt embeddings
            padded_prompt_embeddings = torch.zeros(
                tensordict.batch_size
                + (
                    self.n_steps + 2,
                    new_prompt_embeddings.shape[2],
                ),  # bbox + number of points(steps)
                dtype=torch.float32,
                device=tensordict.device,
            )
            padded_prompt_embeddings[:, : new_prompt_embeddings.shape[1], :] = (
                new_prompt_embeddings
            )
            prompt_embedding_attention_mask = torch.full(
                tensordict.batch_size + (self.n_steps + 2,),
                False,
                dtype=torch.bool,
                device=tensordict.device,
            )
            prompt_embedding_attention_mask[:, : new_prompt_embeddings.shape[1]] = True

            multiclass_padded_prompt_embeddings[instance_idx] = padded_prompt_embeddings
            multiclass_prompt_embedding_attention_masks[instance_idx] = (
                prompt_embedding_attention_mask
            )
            multiclass_image_logits[instance_idx] = new_image_logits
            multiclass_point_coords[instance_idx] = new_point_coords
            multiclass_point_labels[instance_idx] = new_point_labels

        done = torch.full(
            tensordict.batch_size + DONE_SHAPE,
            False,
            dtype=torch.bool,
            device=tensordict.device,
        )
        done[tensordict["step"] + 1 == self.n_steps] = True

        return TensorDict(
            {
                "image": tensordict["image"],
                "image_embedding1": tensordict["image_embedding1"],
                "image_embedding2": tensordict["image_embedding2"],
                "image_embedding3": tensordict["image_embedding3"],
                "image_embedding4": tensordict["image_embedding4"],
                "multiclass_padded_prompt_embeddings": multiclass_padded_prompt_embeddings,
                "multiclass_prompt_embedding_attention_masks": multiclass_prompt_embedding_attention_masks,
                "boxes": tensordict["boxes"],
                "multiclass_image_logits": multiclass_image_logits,
                "multiclass_point_coords": multiclass_point_coords,
                "multiclass_point_labels": multiclass_point_labels,
                "step": tensordict["step"] + 1,
                "true_multiclass_segmentation": tensordict[
                    "true_multiclass_segmentation"
                ],
                "done": done,
                "reward": reward,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )


def get_image_embedder_fn(
    ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device
) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> list[ImageEmbedding]:
        # Ahus model expects a batch and channel dimension
        image_embeddings = ahus_model.image_encoder(
            image.to(ahus_model_device).unsqueeze(0).unsqueeze(0)
        )
        # Move the embeddings to the env device and remove batch dimension
        image_embeddings = [emb.to(env_device).squeeze(0) for emb in image_embeddings]
        return image_embeddings

    return image_embedder_fn


def get_image_logits_fn(
    ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device
) -> ImageLogitsFn:
    def mask_fn(
        image_embeddings: list[ImageEmbedding],
        bbox: Box | None,
        point_coords: PointCoords | None,
        point_labels: PointLabels | None,
        image_logits: ImageLogits | None,
    ) -> tuple[ImageLogits, PromptEmbeddings]:
        # decoder_forward expects everything in batched format
        image_embeddings = [
            emb.to(ahus_model_device).unsqueeze(0) for emb in reversed(image_embeddings)
        ]
        bbox = bbox.to(ahus_model_device).unsqueeze(0) if bbox is not None else None
        point_coords = (
            point_coords.to(ahus_model_device).unsqueeze(0)
            if point_coords is not None
            else None
        )
        point_labels = (
            point_labels.to(ahus_model_device).unsqueeze(0)
            if point_labels is not None
            else None
        )
        image_logits = (
            image_logits.to(ahus_model_device).unsqueeze(0)
            if image_logits is not None
            else None
        )

        # Assume that if either point_coords or point_labels is None, then both are None
        if point_coords is None or point_labels is None:
            points = None
        else:
            points = (point_coords, point_labels)

        mask_logits, prompt_embeddings = decoder_forward(
            ahus_model,
            image_embeddings,
            image_logits,
            points,
            bbox,
        )

        # Move back to env device and remove batch dimension
        mask_logits: ImageLogits = mask_logits.to(env_device).squeeze(0)
        prompt_embeddings: PromptEmbeddings = prompt_embeddings.to(env_device).squeeze(
            0
        )

        return mask_logits, prompt_embeddings

    return mask_fn


def get_post_processing_fn() -> PostProcessingFn:
    def post_processing_fn(
        image: Image,
        image_logits: MulticlassImageLogits,
        logits_to_add: MulticlassImageLogits,
    ) -> MulticlassSegmentation:
        # Simply take the argmax over the instance dimension
        multiclass_segmentation = (
            (image_logits + logits_to_add).argmax(dim=0).to(torch.uint8)
        )
        return multiclass_segmentation

    return post_processing_fn


def get_interaction_fn() -> InteractionFn:
    def interaction_fn(
        seg: MulticlassSegmentation, true_seg: MulticlassSegmentation
    ) -> tuple[PointCoord, PointLabel]:
        point_coord_list, point_label_list = interact(seg, true_seg)
        # Assume that we only receive one point coord and label
        point_coord: PointCoord = torch.cat(point_coord_list, dim=0)
        point_label: PointLabel = torch.cat(point_label_list, dim=0)
        return point_coord, point_label

    return interaction_fn


def get_reward_fn() -> RewardFn:
    # TODO: use step
    def reward_fn(
        seg: MulticlassSegmentation, true_seg: MulticlassSegmentation, step: Step
    ) -> Reward:
        dsc_tensor, nsd_tensor = compute_multi_class_dsc_nsd(
            true_seg,
            seg,
            spacing=torch.ones(seg.shape[0], 3, dtype=torch.float32),
            tolerance=2.0,
        )
        # ~normalized to 0 mean
        return (dsc_tensor + nsd_tensor).unsqueeze(-1) - 1.5

    return reward_fn


def infinite_loader(factory: Callable[[], Iterator]):
    while True:
        iterator = factory()
        for elem in iterator:
            yield elem


def get_env(
    ahus_model: AhusModel,
    ahus_model_device: torch.device,
    env_device: torch.device,
    td_iterator_factory: Callable[[], Iterator[TensorDict]],
):
    env = InteractiveSegmentationEnv(
        n_steps=5,
        image_embedder_fn=get_image_embedder_fn(
            ahus_model, ahus_model_device, env_device=env_device
        ),
        image_logits_fn=get_image_logits_fn(
            ahus_model, ahus_model_device=ahus_model_device, env_device=env_device
        ),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(),
        reward_fn=get_reward_fn(),
        td_iter=infinite_loader(
            td_iterator_factory,
        ),
        device=env_device,
    )
    return env
