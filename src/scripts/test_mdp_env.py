import argparse
from types import SimpleNamespace
from typing import List, Tuple

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

from custom_types import (
    BBox,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    InteractionFn,
    Mask,
    MaskFn,
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
from model.build_ahus_model import build_ahus_model
from model.modeling import AhusModel
from rl.mdp_env import InteractiveSegmentationEnv
from train_ahus_model import get_dataloaders_npz
from utils.decode import decoder_forward
from utils.interact import interact
from utils.postprocess import standard_threshold
from utils.reward import compute_multi_class_dsc_nsd_batch


@jaxtyped(typechecker=beartype)
def get_image_embedder_fn(ahus_model: AhusModel) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> List[ImageEmbedding]:
        return ahus_model.image_encoder(image)

    return image_embedder_fn


@jaxtyped(typechecker=beartype)
def get_mask_fn(ahus_model: AhusModel) -> MaskFn:
    def mask_fn(
        image_embeddings: List[ImageEmbedding], bbox: BBox, points: Points, point_labels: PointLabels, mask: Mask
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
            true_seg, seg, spacing=torch.ones(seg.shape[0], 3, dtype=torch.float32), tolerance=2.0
        )
        return (dsc_tensor + nsd_tensor).unsqueeze(-1)

    return reward_fn


def main():
    device = torch.device("cuda:1")

    ahus_model = build_ahus_model()
    ahus_model = ahus_model.to(device)

    # checkpoint path as arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ahus_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    ahus_model.requires_grad_(False)
    ahus_model.eval()

    batch_size = torch.Size((2,))
    env = InteractiveSegmentationEnv(
        n_steps=5,
        image_embedder_fn=get_image_embedder_fn(ahus_model),
        mask_fn=get_mask_fn(ahus_model),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(),
        reward_fn=get_reward_fn(),
        # only use training data
        dataset_iter=iter(
            get_dataloaders_npz(
                args=SimpleNamespace(
                    base_dir="/home/valter/Desktop/3D_train_npz_random_10percent_16G",
                    val_dir="...",
                    img_size=128,
                    batch_size=batch_size[0],
                    num_workers=4,
                )
            )[0]
        ),
        device=device,
        batch_size=batch_size,
        image_shape=(128, 128, 128),  # TODO: allow for any image shape
    )
    td = env.reset()
    td = env.rand_step(td)
    check_env_specs(env)

    td: TensorDict = env.rollout(10)

    print(td)


if __name__ == "__main__":
    main()
