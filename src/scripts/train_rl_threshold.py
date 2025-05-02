import argparse
from pathlib import Path
import os
from os.path import expandvars

import torch
import wandb
import yaml
# from dotenv import load_dotenv
from monai.transforms.croppad.array import CropForeground
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

from src.dataset.npz_dataset import NPZDataset
from src.model.build_ahus_model import model_registry
from src.rl.agents import THRESHOLD_AGENT_REGISTRY, ThresholdAgent
from src.rl.mdp_env import get_env
from src.rl.utils import dict_to_namespace

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())

# load_dotenv()


def main():
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

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
        transform=CropForeground(select_fn=lambda x: x > 0, k_divisible=8, allow_smaller=True),
        data_suffix="npz",
        label_dtype=torch.long,
    )

    env = get_env(
        ahus_model=ahus_model,
        ahus_model_device=config.ahus_model.device,
        env_device=config.env.device,
        dataset=dataset,
    )

    # Just a test
    td = env.reset()
    td = env.rand_step(td)
    td = env.rollout(100)

    # Static type is ThresholdAgent, dynamic type is chosen by config
    agent_cls = THRESHOLD_AGENT_REGISTRY[config.agent.type]
    agent: ThresholdAgent = agent_cls(
        device=config.agent.device,
        **config_dict["agent"]["kwargs"],
    )

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=1,
        total_frames=config.total_frames,
        storing_device="cpu",  # Since we do logging anyways
        env_device=config.env.device,
        policy_device=config.agent.device,
        trust_policy=True,
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

            loss, grad_norm = agent.process_batch(td.to(config.agent.device))

            # Log network parameter norm
            wandb.log(
                {
                    "reward": td["next", "reward"].item(),
                    "threshold": td["threshold"].item(),
                    "step": td["step"].item(),
                    "loss": loss,
                    "grad_norm": grad_norm,
                    **agent.get_info(),
                }
            )

            if batch_idx % config.eval_every_n_batches == 0:
                # Evaluate the model
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_td = env.rollout(max_steps=config.eval_max_steps, policy=agent.policy)

                    # Move to cpu for logging
                    eval_td = eval_td.to("cpu")

                    # Log evaluation results
                    wandb.log(
                        {
                            "eval reward": eval_td["next", "reward"].mean().item(),
                            "eval threshold": eval_td["threshold"].mean().item(),
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
    main()
