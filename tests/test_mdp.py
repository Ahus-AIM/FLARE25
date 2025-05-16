import torch

from src.rl.utils import (
    image_logits_to_multiclass_segmentation,
    multiclass_to_singleclass_segmentations,
)


def test_image_logits_to_multiclass_segmentation():
    batched_model_logits = torch.tensor(
        [
            [
                [-10, -10, -10],
                [10, 10, -10],
                [10, 10, -10],
            ],
            [
                [-10, -10, -10],
                [-10, 8, 10],
                [-10, 8, 10],
            ],
        ],
        dtype=torch.float32,
    )

    expected_multiclass_segmentation = torch.tensor(
        [
            [
                [0, 0, 0],
                [1, 1, 2],
                [1, 1, 2],
            ],
        ],
        dtype=torch.uint8,
    )

    actual_multiclass_segmentation = image_logits_to_multiclass_segmentation(
        batched_model_logits,
    )

    assert torch.allclose(
        actual_multiclass_segmentation,
        expected_multiclass_segmentation,
    )


def test_multiclass_to_singleclass_segmentations():
    multiclass_segmentation = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 2],
            [1, 1, 2],
        ],
        dtype=torch.long,
    )

    expected_singleclass_segmentations = torch.tensor(
        [
            [
                [0, 0, 0],
                [1, 1, 0],
                [1, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
        ],
        dtype=torch.bool,
    )

    actual_singleclass_segmentations = multiclass_to_singleclass_segmentations(
        multiclass_segmentation,
        n_classes=2,
    )

    assert torch.allclose(
        actual_singleclass_segmentations,
        expected_singleclass_segmentations,
    )


# OUTDATED

# @jaxtyped(typechecker=beartype)
# def get_image_embedder_fn(ahus_model: AhusModel) -> ImageEmbedderFn:
#     def image_embedder_fn(image: Image) -> List[ImageEmbedding]:
#         return ahus_model.image_encoder(image)

#     return image_embedder_fn


# @jaxtyped(typechecker=beartype)
# def get_mask_fn(ahus_model: AhusModel) -> ImageLogitsFn:
#     def mask_fn(
#         image_embeddings: List[ImageEmbedding],
#         bbox: MulticlassBoxes,
#         points: PointCoords,
#         point_labels: PointLabels,
#         mask: ImageLogits,
#     ) -> ImageLogits:
#         # TODO: this looks bad
#         image_embeddings = image_embeddings[::-1]
#         return decoder_forward(ahus_model, image_embeddings, mask, (points, point_labels), bbox)[0]

#     return mask_fn


# @jaxtyped(typechecker=beartype)
# def get_post_processing_fn() -> PostProcessingFn:
#     def post_processing_fn(image: Image, mask: ImageLogits, threshold: Threshold) -> MulticlassSegmentation:
#         return standard_threshold(mask, threshold)

#     return post_processing_fn


# @jaxtyped(typechecker=beartype)
# def get_interaction_fn() -> InteractionFn:
#     def interaction_fn(seg: MulticlassSegmentation, true_seg: MulticlassSegmentation) -> Tuple[PointCoord, PointLabel]:
#         point_list, point_label_list = interact(seg, true_seg)
#         # Assume that we only get a single point
#         point_coord: PointCoord = torch.cat(point_list, dim=0)
#         point_label: PointLabel = torch.cat(point_label_list, dim=0)
#         return point_coord, point_label

#     return interaction_fn


# @jaxtyped(typechecker=beartype)
# def get_reward_fn() -> RewardFn:
#     # TODO: use step
#     def reward_fn(seg: MulticlassSegmentation, true_seg: MulticlassSegmentation, step: Step) -> Reward:
#         dsc_tensor, nsd_tensor = compute_multi_class_dsc_nsd_batch(
#             true_seg,
#             seg,
#             spacing=torch.ones(seg.shape[0], 3, dtype=torch.float32),
#             tolerance=2.0,
#         )
#         return (dsc_tensor + nsd_tensor).unsqueeze(-1)

#     return reward_fn


# class MockDatasetIter(Iterator[MedicalData]):
#     def __init__(self):
#         self.depth_options = [16, 32, 64, 128]
#         self.height_options = [32, 64, 128, 256]
#         self.width_options = [64, 128, 256, 512]
#         self.n_options = len(self.depth_options)
#         self.size_idx = 0  # What the size of the next image will be

#     def __iter(self):
#         return self

#     def __next__(self):
#         depth: int = self.depth_options[self.size_idx]
#         height: int = self.height_options[self.size_idx]
#         width: int = self.width_options[self.size_idx]
#         self.size_idx = (self.size_idx + 1) % self.n_options
#         image: Image = torch.rand(1, 1, depth, height, width)
#         boxes: MulticlassBoxes = torch.rand(1, 2, 3)
#         label: MulticlassSegmentation = torch.randint(0, 2, (1, 1, depth, height, width), dtype=torch.int64)
#         medical_data: MedicalData = {
#             "image": image,
#             "boxes": boxes,
#             "label": label,
#         }
#         return medical_data


# def test_mdp_no_errors():
#     """
#     Tests whether the MDP environment can be created without errors. Assumes that the data path is stored in the
#     environment variable MEDSEG_DATA_PATH.
#     """
#     cp.cuda.set_allocator(None)
#     device = torch.device("cpu")

#     ahus_model = model_registry["ahus_model_liere"]()
#     ahus_model = ahus_model.to(device)

#     # We don't load a checkpoint since we don't care about performance for this test

#     ahus_model.requires_grad_(False)
#     ahus_model.eval()

#     # TODO: env does not work with n_steps != 5
#     env = InteractiveSegmentationEnv(
#         n_steps=5,
#         image_embedder_fn=get_image_embedder_fn(ahus_model),
#         image_logits_fn=get_mask_fn(ahus_model),
#         post_processing_fn=get_post_processing_fn(),
#         interaction_fn=get_interaction_fn(),
#         reward_fn=get_reward_fn(),
#         td_iter=MockDatasetIter(),
#         device=device,
#     )
#     check_env_specs(env)

#     # For the first episode, images should have shape (1, 32, 64, 128)
#     td = env.reset()
#     assert td["image"].shape == (1, 1, 32, 64, 128)
#     assert td["mask"].shape == (1, 1, 32, 64, 128)
#     assert td["true_segmentation"].shape == (1, 1, 32, 64, 128)
#     # step should work
#     td = env.rand_step(td)

#     # For the second episode, images should have shape (1, 64, 128, 256)
#     td = env.reset()
#     assert td["image"].shape == (1, 1, 64, 128, 256)
#     assert td["mask"].shape == (1, 1, 64, 128, 256)
#     assert td["true_segmentation"].shape == (1, 1, 64, 128, 256)
#     # step should work
#     td = env.rand_step(td)

#     # For the third episode, images should have shape (1, 128, 256, 512)
#     td = env.reset()
#     assert td["image"].shape == (1, 1, 128, 256, 512)
#     assert td["mask"].shape == (1, 1, 128, 256, 512)
#     assert td["true_segmentation"].shape == (1, 1, 128, 256, 512)
#     # step should work
#     td = env.rand_step(td)


# if __name__ == "__main__":
#     test_mdp_no_errors()
