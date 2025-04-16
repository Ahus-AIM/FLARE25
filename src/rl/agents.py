from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Bounded
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss


class PPOThresholdAgent:
    def __init__(self, device: torch.device, lr: float, clip_epsilon=0.2, entropy_bonus=False, max_grad_norm=1.0):
        self.device = device
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_bonus = entropy_bonus
        self.max_grad_norm = max_grad_norm

        # Define the actor network
        self.actor_net = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(3, 3, 3)),  # (N, 1, 3, 3, 3)
            nn.Flatten(start_dim=1),  # (N, 27)
            nn.Linear(27, 2),  # Let autocast handle dtype
            NormalParamExtractor(),
        ).to(self.device)

        policy_module = TensorDictModule(self.actor_net, in_keys=["mask"], out_keys=["loc", "scale"])

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=Bounded(low=0, high=1, shape=torch.Size(())),  # Avoid setting float16 here
            in_keys=["loc", "scale"],
            out_keys=["threshold"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": 0, "high": 1},
            return_log_prob=True,
        )

        # Define the value network
        self.value_net = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(3, 3, 3)), nn.Flatten(start_dim=1), nn.Linear(27, 1)
        ).to(self.device)

        self.value_module = ValueOperator(module=self.value_net, in_keys=["mask"])

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
        )
        self.loss_keys = ["loss_objective", "loss_critic"]
        if self.entropy_bonus:
            self.loss_keys.append("loss_entropy")
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        # You can wrap this in autocast when calling from training loop if needed
        return self.policy_module(td)

    def process_batch(self, td: TensorDictBase):
        """
        Process a batch of data, updating value networks and policy
        """
        # Zero the gradients
        self.optim.zero_grad()

        # Compute the loss
        loss_td = self.loss_module(td)

        # loss = loss_td["loss_objective"] + loss_td["loss_critic"]
        loss: Any = sum([loss_td[k] for k in self.loss_keys])

        # Backpropagation
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        # Update the policy and value networks
        self.optim.step()

        return loss, grad_norm

    def save(self, path: str) -> None:
        """
        Save the model weights and hyperparameters to a single file.

        Args:
            path (str): Path to the file where the model and hyperparameters will be saved.
        """
        # Prepare the data to save
        checkpoint = {
            "state_dict": self.policy_module.state_dict(),
            "hyperparameters": {
                "lr": self.lr,
                "clip_epsilon": self.clip_epsilon,
                "entropy_bonus": self.entropy_bonus,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        # Save everything in one file
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "PPOThresholdAgent":
        """
        Load the policy model weights and hyperparameters from a single file.

        Args:
            path (str): Path to the file containing the model and hyperparameters
            device (torch.device): Device to load the model on
        """
        # Load the combined checkpoint
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["hyperparameters"]

        # Create the agent using the loaded hyperparameters
        agent = PPOThresholdAgent(
            device=device,
            lr=config["lr"],
            clip_epsilon=config["clip_epsilon"],
            entropy_bonus=config["entropy_bonus"],
            max_grad_norm=config["max_grad_norm"],
        )

        # Load the model weights
        agent.policy_module.load_state_dict(checkpoint["state_dict"])

        return agent
