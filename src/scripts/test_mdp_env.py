from types import SimpleNamespace
from typing import Tuple

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from tensordict import TensorDict  # type: ignore
from torchrl.envs.utils import check_env_specs  # type: ignore

from src.custom_types import (
    BBox,
    Image,
    ImageEmbedderFn,
    ImageEmbedding,
    InteractionFn,
    LowresMask,
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
from src.model.build_sam3D import sam_model_registry3D  # type: ignore
from src.model.modeling import Sam3D
from src.rl.mdp_env import InteractiveSegmentationEnv
from src.train import get_dataloaders_npz  # type: ignore
from src.utils.decode import decode_batch_low_res
from src.utils.interact import interact
from src.utils.postprocess import trilinear_upsample_threshold
from src.utils.reward import compute_multi_class_dsc_nsd_batch


@jaxtyped(typechecker=beartype)
def get_image_embedder_fn(sam_model: Sam3D) -> ImageEmbedderFn:
    def image_embedder_fn(image: Image) -> ImageEmbedding:
        return sam_model.image_encoder(image)

    return image_embedder_fn


@jaxtyped(typechecker=beartype)
def get_mask_fn(sam_model: Sam3D) -> MaskFn:
    def mask_fn(
        image_embedding: ImageEmbedding, bbox: BBox, points: Points, point_labels: PointLabels, low_res_mask: LowresMask
    ) -> LowresMask:
        return decode_batch_low_res(sam_model, image_embedding, low_res_mask, points, point_labels, bbox)

    return mask_fn


@jaxtyped(typechecker=beartype)
def get_post_processing_fn() -> PostProcessingFn:
    def post_processing_fn(image: Image, low_res_mask: LowresMask, threshold: Threshold) -> Segmentation:
        return trilinear_upsample_threshold(low_res_mask, image, threshold)

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

    sam_model = sam_model_registry3D["vit_b_ori_norm"](checkpoint=None).to(device)

    ckpt = torch.load(  # type: ignore
        "/home/valter/Desktop/valter_ckpt_not_converged_10-03-2025.pth", map_location=device, weights_only=False
    )

    sam_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    sam_model.requires_grad_(False)
    sam_model.eval()

    batch_size = torch.Size((2,))
    env = InteractiveSegmentationEnv(
        n_steps=5,
        image_embedder_fn=get_image_embedder_fn(sam_model),
        mask_fn=get_mask_fn(sam_model),
        post_processing_fn=get_post_processing_fn(),
        interaction_fn=get_interaction_fn(),
        reward_fn=get_reward_fn(),
        dataset_iter=iter(
            get_dataloaders_npz(
                args=SimpleNamespace(
                    base_dir="/home/valter/Desktop/3D_train_npz_random_10percent_16G",
                    img_size=128,
                    batch_size=batch_size[0],
                    num_workers=4,
                )
            )  # type: ignore
        ),  # type: ignore
        device=device,
        batch_size=batch_size,
    )
    check_env_specs(env)

    td: TensorDict = env.rollout(10)

    print(td)


if __name__ == "__main__":
    main()
