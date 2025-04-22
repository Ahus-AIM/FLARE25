import argparse

import torch

from src.rl.agents import PPOThresholdAgent


def main(args: argparse.Namespace) -> None:
    agent = PPOThresholdAgent(
        device=torch.device("cpu"),
        lr=0.0001,
        clip_epsilon=0.2,
        entropy_bonus=False,
        max_grad_norm=0.5,
    )
    agent.save(args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dummy agent checkpoint.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to save the dummy agent checkpoint.",
    )
    args = parser.parse_args()
    main(args)
