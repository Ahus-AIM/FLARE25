import argparse
import os

import torch

from rl.agents.attention_based.ppo import PPOThresholdAgent


def main(args: argparse.Namespace) -> None:
    agent = PPOThresholdAgent(
        device=torch.device("cpu"),
        lr=0.0001,
        clip_epsilon=0.2,
        entropy_bonus=False,
        max_grad_norm=0.5,
    )
    # Make parent directory if it doesn't exist
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    agent.save(args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dummy agent checkpoint.")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to save the dummy agent checkpoint.",
    )
    args = parser.parse_args()
    main(args)
