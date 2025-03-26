from typing import Iterator, List, Tuple

import cupy as cp
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torchrl.envs import check_env_specs

from rl.mdp_env import InteractiveSegmentationEnv
from src.custom_types import (
    BBox,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    InteractionFn,
    Mask,
    MaskFn,
    MedicalData,
    Point,
    PointLabel,
    PointLabels,
    Points,
    PostProcessingFn,
    Reward,
    RewardFn,
    Segmentation,
    Step,
    Threshold,
)
from src.model.build_ahus_model import build_ahus_model
from src.model.modeling import AhusModel
from src.utils.decode import decoder_forward
from src.utils.interact import interact
from src.utils.postprocess import standard_threshold
from src.utils.reward import compute_multi_class_dsc_nsd_batch


@jaxtyped(typechecker=beartype)
def get_image_embedder_fn(ahus_model: AhusModel) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> List[ImageEmbedding]:
        return ahus_model.image_encoder(image)

    return image_embedder_fn


@jaxtyped(typechecker=beartype)
def get_mask_fn(ahus_model: AhusModel) -> MaskFn:
    def mask_fn(
        image_embeddings: List[ImageEmbedding],
        bbox: BBox,
        points: Points,
        point_labels: PointLabels,
        mask: Mask,
    ) -> Mask:
        # TODO: this looks bad
        image_embeddings = image_embeddings[::-1]
        return decoder_forward(ahus_model, image_embeddings, mask, (points, point_labels), bbox)

    return mask_fn


@jaxtyped(typechecker=beartype)
def get_post_processing_fn() -> PostProcessingFn:
    def post_processing_fn(image: Image, mask: Mask, threshold: Threshold) -> Segmentation:
        return standard_threshold(mask, threshold)

    return post_processing_fn


@jaxtyped(typechecker=beartype)
def get_interaction_fn() -> InteractionFn:
    def interaction_fn(seg: Segmentation, true_seg: Segmentation) -> Tuple[Point, PointLabel]:
        point_list, point_label_list = interact(seg, true_seg)
        point: Point = torch.cat(point_list, dim=0)
        point_label: PointLabel = torch.cat(point_label_list, dim=0)
        return point, point_label

    return interaction_fn


@jaxtyped(typechecker=beartype)
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


class MockDatasetIter(Iterator[MedicalData]):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __iter(self):
        return self

    def __next__(self):
        image: Image = torch.rand(self.batch_size, 1, 128, 128, 128)
        boxes: BBox = torch.rand(self.batch_size, 2, 3)
        label: Segmentation = torch.randint(0, 2, (self.batch_size, 1, 128, 128, 128), dtype=torch.int64)
        medical_data: MedicalData = {
            "image": image,
            "boxes": boxes,
            "label": label,
        }
        return medical_data


def test_mdp_no_errors():
    """
    Tests whether the MDP environment can be created without errors. Assumes that the data path is stored in the
    environment variable MEDSEG_DATA_PATH.
    """
    cp.cuda.set_allocator(None)
    device = torch.device("cpu")

    ahus_model = build_ahus_model()
    ahus_model = ahus_model.to(device)

    # We don't load a checkpoint since we don't care about performance for this test

    ahus_model.requires_grad_(False)
    ahus_model.eval()

    batch_size = 1
    env = InteractiveSegmentationEnv(
        n_steps=3,
        image_embedder_fn=get_image_embedder_fn(ahus_model),
        mask_fn=get_mask_fn(ahus_model),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(),
        reward_fn=get_reward_fn(),
        # only use training data
        dataset_iter=MockDatasetIter(batch_size=batch_size),
        device=device,
        batch_size=torch.Size((batch_size,)),
        image_shape=(128, 128, 128),  # TODO: allow for any image shape
    )
    td = env.reset()
    td = env.rand_step(td)
    check_env_specs(env)


if __name__ == "__main__":
    test_mdp_no_errors()
