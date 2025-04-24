from typing import Iterator

import torch
from beartype.typing import List, Tuple
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs import EnvBase

from src.custom_types import (
    BBOX_SHAPE,
    DONE_SHAPE,
    POINT_COORD_SHAPE,
    POINT_LABEL_SHAPE,
    REWARD_SHAPE,
    STEP_SHAPE,
    THRESHOLD_SHAPE,
    BBox,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    InteractionFn,
    Mask,
    MaskFn,
    MedicalData,
    PointCoord,
    PointCoords,
    PointLabel,
    PointLabels,
    PostProcessingFn,
    Reward,
    RewardFn,
    Segmentation,
    Step,
    Threshold,
)
from src.dataset.npz_dataset import NPZDataset
from src.model.modeling import AhusModel
from src.utils.decode import decoder_forward
from src.utils.interact import interact
from src.utils.postprocess import standard_threshold
from src.utils.reward import compute_multi_class_dsc_nsd_batch


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
            dataset_iter: iterator over the dataset with batch size 1
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
                shape=self.batch_size + BBOX_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            # mask has a single channel, same shape as image
            mask=Unbounded(
                shape=self.batch_size + (1, -1, -1, -1),
                dtype=torch.float32,
                domain="continuous",
            ),
            # point_coords and point_labels are actually bounded, but we don't know the image size beforehand
            point_coords=Unbounded(
                shape=self.batch_size + (self.n_steps,) + POINT_COORD_SHAPE,
                dtype=torch.float32,
                domain="continuous",
            ),
            point_labels=Unbounded(
                shape=self.batch_size + (self.n_steps,) + POINT_LABEL_SHAPE,
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
                shape=self.batch_size + THRESHOLD_SHAPE,
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
                "point_coords": torch.zeros(
                    tensordict.batch_size + (self.n_steps,) + POINT_COORD_SHAPE,
                    dtype=torch.float32,
                    device=tensordict.device,
                ),
                "point_labels": torch.zeros(
                    tensordict.batch_size + (self.n_steps,) + POINT_LABEL_SHAPE,
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
                "step": torch.zeros(tensordict.batch_size + STEP_SHAPE, dtype=torch.int64, device=tensordict.device),
                "true_segmentation": true_segmentation,
                "done": torch.full(
                    tensordict.batch_size + DONE_SHAPE,
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
            tensordict["point_coords"][..., :step, :],
            tensordict["point_labels"][..., :step],
            tensordict["mask"],
        )

        # Always use upsampling method 0 for now
        segmentation = self.post_processing_fn(tensordict["image"], new_low_res_mask, tensordict["threshold"])

        new_point_coord, new_point_label = self.interaction_fn(segmentation, tensordict["true_segmentation"])
        # add new point coord and new label to the point_coords and point_labels tensors
        new_point_coords = tensordict["point_coords"].clone()
        new_point_coords[:, [step], :] = new_point_coord
        new_point_labels = tensordict["point_labels"].clone()
        new_point_labels[:, [step]] = new_point_label

        reward = self.reward_fn(segmentation, tensordict["true_segmentation"], tensordict["step"])

        done = torch.full(tensordict.batch_size + DONE_SHAPE, False, dtype=torch.bool, device=self.device)
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
                "point_coords": new_point_coords,
                "point_labels": new_point_labels,
                "step": tensordict["step"] + 1,
                "true_segmentation": tensordict["true_segmentation"],
                "done": done,
                "reward": reward,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )


def get_image_embedder_fn(
    ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device
) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> List[ImageEmbedding]:
        image_embeddings = ahus_model.image_encoder(image.to(ahus_model_device))
        # Move the embeddings to the env device
        image_embeddings = [emb.to(env_device) for emb in image_embeddings]
        return image_embeddings

    return image_embedder_fn


def get_mask_fn(ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device) -> MaskFn:
    def mask_fn(
        image_embeddings: List[ImageEmbedding],
        bbox: BBox,
        point_coords: PointCoords,
        point_labels: PointLabels,
        mask: Mask,
    ) -> Mask:
        # TODO: this looks bad
        image_embeddings = image_embeddings[::-1]
        image_embeddings = [emb.to(ahus_model_device) for emb in image_embeddings]
        return decoder_forward(
            ahus_model,
            image_embeddings,
            mask.to(ahus_model_device),
            (point_coords.to(ahus_model_device), point_labels.to(ahus_model_device)),
            bbox.to(ahus_model_device),
        ).to(env_device)

    return mask_fn


def get_post_processing_fn() -> PostProcessingFn:
    def post_processing_fn(image: Image, mask: Mask, threshold: Threshold) -> Segmentation:
        return standard_threshold(mask, threshold)

    return post_processing_fn


def get_interaction_fn() -> InteractionFn:
    def interaction_fn(seg: Segmentation, true_seg: Segmentation) -> Tuple[PointCoord, PointLabel]:
        point_coord_list, point_label_list = interact(seg, true_seg)
        # Assume that we only receive one point coord and label
        point_coord: PointCoord = torch.cat(point_coord_list, dim=0)
        point_label: PointLabel = torch.cat(point_label_list, dim=0)
        return point_coord, point_label

    return interaction_fn


def get_reward_fn() -> RewardFn:
    # TODO: use step
    def reward_fn(seg: Segmentation, true_seg: Segmentation, step: Step) -> Reward:
        dsc_tensor, nsd_tensor = compute_multi_class_dsc_nsd_batch(
            true_seg,
            seg,
            spacing=torch.ones(seg.shape[0], 3, dtype=torch.float32),
            tolerance=2.0,
        )
        return (dsc_tensor + nsd_tensor).unsqueeze(-1)

    return reward_fn


def infinite_loader(dataset, batch_size=1, shuffle=True):
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for batch in loader:
            yield batch


def get_env(ahus_model: AhusModel, ahus_model_device: torch.device, env_device: torch.device, dataset: NPZDataset):
    env = InteractiveSegmentationEnv(
        n_steps=5,
        image_embedder_fn=get_image_embedder_fn(ahus_model, ahus_model_device, env_device=env_device),
        mask_fn=get_mask_fn(ahus_model, ahus_model_device=ahus_model_device, env_device=env_device),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(),
        reward_fn=get_reward_fn(),
        dataset_iter=infinite_loader(
            dataset,
            batch_size=1,
            shuffle=True,
        ),
        device=env_device,
    )
    return env
