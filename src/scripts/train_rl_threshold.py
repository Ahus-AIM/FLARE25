import argparse
import os
from pathlib import Path

import torch
import yaml
from monai.transforms.croppad.array import CropForeground
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    ExcludeTransform,
    ExplorationType,
    check_env_specs,
    set_exploration_type,
)
from tqdm import tqdm

import wandb
from src.rl.agents import Agent
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent
from src.dataset.npz_dataset import NPZDataset
from src.model.build_ahus_model import model_registry
from src.rl.agents.attention_based.ddpg import AttentionDDPGThresholdAgent
from src.rl.mdp_env import get_env
from src.rl.utils import dict_to_namespace


def main(args: argparse.Namespace) -> None:
    # Load config from yaml file
    with open(args.config, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    ahus_model = model_registry[config.ahus_model.type]()
    ahus_model = ahus_model.to(config.ahus_model.device)

    # Load checkpoint, assuming it is stored in the "weights" directory
    ckpt = torch.load(
        config.ahus_model.weights_path,
        map_location=config.ahus_model.device,
        weights_only=False,
    )
    ahus_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    ahus_model.eval()
    ahus_model.requires_grad_(False)

    dataset = NPZDataset(
        base_dir=config.dataset.base_dir,
        size_threshold=config.dataset.size_threshold,
        transform=CropForeground(
            select_fn=lambda x: x > 0, k_divisible=8, allow_smaller=True
        ),
        data_suffix="npz",
        label_dtype=torch.long,
    )

    env = get_env(
        ahus_model=ahus_model,
        ahus_model_device=config.ahus_model.device,
        env_device=config.env.device,
        dataset=dataset,
    )
    check_env_specs(env)

    # Get image, should always be the same
    # image1 = env.reset()["image"].cpu()
    # image2 = env.reset()["image"].cpu()
    # assert torch.equal(image1, image2), "Image should always be the same"

    # agent: Agent = AttentionDDPGThresholdAgent(
    #     action_spec=env.action_spec,
    #     **config_dict["agent"]["kwargs"],
    # )

    agent: Agent = AttentionPPOThresholdAgent(
        action_spec=env.action_spec,
        **config_dict["agent"]["kwargs"],
    )

    # We want to compare our agent to a baseline policy of always choosing the threshold 0.5
    def baseline_policy(td: TensorDictBase) -> TensorDictBase:
        # Always choose the threshold 0.5
        td["threshold"] = 0.5 * torch.ones(td.batch_size + (1,), device=td.device)
        return td

    # Debug manually
    td = env.reset()
    td = agent.policy(td)
    td = env.step(td)
    td = td["next"]
    td = agent.policy(td)
    td = env.step(td)

    keys_to_exclude = [
        "bbox",
        "collector",
        "image",
        "image_embedding1",
        "image_embedding2",
        "image_embedding3",
        "image_embedding4",
        "mask",
        "point_coords",
        "point_labels",
        "true_segmentation",
        # next
        ("next", "bbox"),
        ("next", "image"),
        ("next", "image_embedding1"),
        ("next", "image_embedding2"),
        ("next", "image_embedding3"),
        ("next", "image_embedding4"),
        ("next", "mask"),
        ("next", "point_coords"),
        ("next", "point_labels"),
        ("next", "true_segmentation"),
    ]
    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=1,
        total_frames=config.total_frames,
        # storing_device="cpu",  # Since we do logging anyways
        # env_device=env.device,
        # policy_device=agent.device,
        # trust_policy=True,
        postproc=ExcludeTransform(*keys_to_exclude),
    )

    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=config_dict,
    )

    try:
        for batch_idx, td in tqdm(enumerate(collector), total=config.total_frames):
            # Agent and batch dimension are collapsed
            td = td.reshape(-1, *td.shape[2:])

            loss_info = agent.process_batch(td.to(agent.device))

            # Log network parameter norm
            wandb.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        {
                            "reward": td["next", "reward"].item(),
                            "threshold": td["threshold"].item(),
                            "step": td["step"].item(),
                        }
                        | loss_info
                        | agent.get_train_info()
                    ).items()
                }
            )

            if batch_idx % config.eval_every_n_batches == 0:
                tqdm.write("Evaluating...")
                # Evaluate the model
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_td = env.rollout(
                        max_steps=config.eval_max_steps, policy=agent.policy
                    )
                    baseline_td = env.rollout(
                        max_steps=config.eval_max_steps, policy=baseline_policy
                    )

                    # Move to cpu for logging
                    eval_td = eval_td.to("cpu")
                    baseline_td = baseline_td.to("cpu")

                    # Log evaluation results
                    wandb.log(
                        {
                            f"eval/{k}": v
                            for k, v in (
                                {
                                    "reward sum": eval_td["next", "reward"]
                                    .sum()
                                    .item(),
                                    "threshold": eval_td["threshold"]
                                    .mean()
                                    .item(),
                                    "baseline reward sum": baseline_td["next", "reward"]
                                    .sum()
                                    .item(),
                                    "baseline threshold": baseline_td["threshold"]
                                    .mean()
                                    .item(),
                                } | agent.get_eval_info()
                            ).items()
                        }
                    )
    except KeyboardInterrupt:
        print("Training interrupted.")

    # Save model
    print(f"Saving model to {config.agent.save_path}...", end="")
    Path(config.agent.save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(Path(config.agent.save_path))
    print("done.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_attention_ppo_threshold.yml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    main(args)
