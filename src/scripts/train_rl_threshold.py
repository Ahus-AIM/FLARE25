import argparse
import time
from pathlib import Path

import torch
import yaml
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (  # check_env_specs,
    ExcludeTransform,
    ExplorationType,
    set_exploration_type,
)
from tqdm import tqdm

import wandb
from src.dataset.tensordict_npz_dataset import (
    get_td_iterator,
)
from src.model.registry import model_registry
from src.rl.agents import Agent
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent
from src.rl.mdp_env import get_env
from src.rl.utils import dict_to_namespace


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(True)

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

    train_env = get_env(
        ahus_model=ahus_model,
        ahus_model_device=config.ahus_model.device,
        env_device=config.env.device,
        td_iterator_factory=lambda: get_td_iterator(
            val_dir=Path(config.dataset.train_dir),
            val_gt_dir=Path(config.dataset.gt_dir),
        ),
        size_threshold=config.size_threshold,
        interact_device=config.env.interact_device,
    )
    eval_envs = [
        get_env(
            ahus_model=ahus_model,
            ahus_model_device=config.ahus_model.device,
            env_device=config.env.device,
            td_iterator_factory=lambda: get_td_iterator(
                val_dir=Path(config.dataset.val_dir),
                val_gt_dir=Path(config.dataset.gt_dir),
            ),
            size_threshold=config.size_threshold,
            interact_device=config.env.interact_device,
        )
        for _ in range(2)
    ]

    # Debug env manually
    # td = train_env.reset()
    # td = train_env.rand_step(td)
    # td = td["next"]
    # td = train_env.rand_step(td)
    # check_env_specs(train_env)

    # Get image, should always be the same
    # image1 = env.reset()["image"].cpu()
    # image2 = env.reset()["image"].cpu()
    # assert torch.equal(image1, image2), "Image should always be the same"

    # agent: Agent = AttentionDDPGThresholdAgent(
    #     action_spec=env.action_spec,
    #     **config_dict["agent"]["kwargs"],
    # )

    agent: Agent = AttentionPPOThresholdAgent(
        action_spec=train_env.action_spec,
        **config_dict["agent"]["kwargs"],
    )

    # We want to compare our agent to a baseline policy of never adding any logits
    def baseline_noop_policy(td: TensorDictBase) -> TensorDictBase:
        # Always choose the threshold 0.5
        td["logits_to_add"] = torch.zeros(
            (*td["downsampled_image_logits"].shape[:-3], 1),
            dtype=torch.float32,
            device=td.device,
        )
        return td

    # We want to compare our agent to a baseline policy of adding too much logits
    def baseline_positive_aggressive_policy(td: TensorDictBase) -> TensorDictBase:
        # Always choose the threshold 0.5
        td["logits_to_add"] = 10 * torch.ones(
            (*td["downsampled_image_logits"].shape[:-3], 1),
            dtype=torch.float32,
            device=td.device,
        )
        return td

    # We want to compare our agent to a baseline policy of adding too much logits
    def baseline_5_policy(td: TensorDictBase) -> TensorDictBase:
        # Always choose the threshold 0.5
        td["logits_to_add"] = 5 * torch.ones(
            (*td["downsampled_image_logits"].shape[:-3], 1),
            dtype=torch.float32,
            device=td.device,
        )
        return td

    # Debug manually
    td = train_env.reset()
    td = agent.policy(td.to(agent.device)).to(train_env.device)
    td = train_env.step(td)
    # td = td["next"]
    # td = agent.policy(td.to(agent.device)).to(train_env.device)
    # td = train_env.step(td)
    # agent.process_batch(td.to(agent.device))

    env_keys_to_exclude = [
        "downsampled_image",
        "image_embedding1",
        "image_embedding2",
        "image_embedding3",
        "image_embedding4",
        # "padded_prompt_embeddings",
        # "prompt_embedding_attention_masks",
        "downsampled_boxes",
        "downsampled_image_logits",
        "downsampled_point_coords",
        "point_labels",
        "true_multiclass_segmentation",
    ]
    keys_to_exclude = [
        "collector",
        *env_keys_to_exclude,
        *(("next", k) for k in env_keys_to_exclude),
    ]
    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=1,
        total_frames=config.total_frames,
        storing_device="cpu",  # Since we do logging anyways
        env_device=train_env.device,
        policy_device=agent.device,
        # trust_policy=True,
        postproc=ExcludeTransform(*keys_to_exclude),
    )

    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=config_dict,
    )

    wandb.watch(agent.backbone_net, log="gradients", log_freq=1000)
    wandb.watch(agent.actor_net, log="gradients", log_freq=1000)
    wandb.watch(agent.value_net, log="gradients", log_freq=1000)

    try:
        for batch_idx, td in tqdm(enumerate(collector), total=config.total_frames):
            # Collapse agent and env batch dimensions
            td = td.flatten(0, 1)
            loss_info = agent.process_batch(td.to(agent.device))

            # Log network parameter norm
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        {
                            "reward": td["next", "reward"].item(),
                            "step": td["step"].item(),
                        }
                        | loss_info
                        | agent.get_train_info()
                    ).items()
                }
            )

            if batch_idx != 0 and batch_idx % config.eval_every_n_batches == 0:
                tqdm.write("Evaluating...")
                # Evaluate the model
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_td = eval_envs[0].rollout(
                        max_steps=config.eval_max_steps,
                        policy=agent.policy,
                        auto_cast_to_device=True,
                    )
                    # eval_td = eval_td.to("cpu")
                    baseline_noop_td = eval_envs[1].rollout(
                        max_steps=config.eval_max_steps,
                        policy=baseline_noop_policy,
                        auto_cast_to_device=True,
                    )
                    # baseline_noop_td = baseline_noop_td.to("cpu")
                    # baseline_positive_aggressive_td = eval_env.rollout(
                    #     max_steps=config.eval_max_steps, policy=baseline_positive_aggressive_policy, auto_cast_to_device=True
                    # )
                    # baseline_positive_aggressive_td = baseline_positive_aggressive_td.to("cpu")
                    # baseline_5_td = eval_env.rollout(
                    #     max_steps=config.eval_max_steps, policy=baseline_5_policy, auto_cast_to_device=True
                    # )
                    # baseline_negative_aggressive_td = baseline_negative_aggressive_td.to("cpu")

                    # Log evaluation results
                    run.log(
                        {
                            f"eval/{k}": v
                            for k, v in (
                                {
                                    "reward sum": eval_td["next", "reward"].cpu().sum().item(),
                                    "baseline noop reward sum": baseline_noop_td["next", "reward"].cpu().sum().item(),
                                    # "baseline positive aggressive reward sum": baseline_positive_aggressive_td["next", "reward"].sum().item(),
                                    # "baseline 5 reward sum": baseline_5_td["next", "reward"].sum().item(),
                                    "logits_to_add": wandb.Histogram(eval_td["logits_to_add"].cpu()),
                                }
                                | agent.get_eval_info()
                            ).items()
                        }
                    )
                # Save model at the end of evaluation
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                agent.save(Path(config.agent.save_path) / timestamp)
    except KeyboardInterrupt:
        print("Training interrupted.")

    # Save model
    print(f"Saving model to {config.agent.save_path}...", end="")
    agent.save(Path(config.agent.save_path) / "latest")
    print("done.")
    run.finish()


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
