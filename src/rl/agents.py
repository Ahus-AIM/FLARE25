from typing import Protocol, Type

from rich.prompt import Prompt

from src.rl.models import PromptAttentionNet
from src.model.modeling.common import normalize
from src.model.modeling.normalized_mask_decoder3D import (
    Attention,
    MLPBlock3D,
)

import torch
from pathlib import Path
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.data import Bounded, ListStorage, TensorDictReplayBuffer
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate

from src.rl.utils import calculate_norm


class ThresholdAgent(Protocol):
    def policy(self, td: TensorDictBase) -> TensorDictBase:
        """
        Updates the 'threshold' key in the TensorDict.
        """
        ...

    def process_batch(self, td: TensorDictBase) -> tuple[float, float]:
        """
        Process a batch of data. Returns the mean loss and gradient norm.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Save everything necessary to recreate the agent, including the model weights and hyperparameters.
        """
        ...

    @staticmethod
    def load(path: Path, device: torch.device) -> "ThresholdAgent":
        """
        Recreate the agent from a saved file.
        """
        ...

    def get_info(self) -> dict:
        """
        Returns a dictionary with information about the agent, like weight norm. Used in logging during training.
        """
        ...


class PPOThresholdAgent(ThresholdAgent):
    def __init__(self, device: torch.device, lr: float, clip_epsilon=0.2, entropy_bonus=False, max_grad_norm=1.0):
        self.device = device
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_bonus = entropy_bonus
        self.max_grad_norm = max_grad_norm

        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (N, 16, D/2, H/2, W/2)
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (N, 32, D/4, H/4, W/4)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (N, 64, D/8, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),  # -> (N, 64, 1, 1, 1)
            nn.Flatten(start_dim=1),  # -> (N, 64)
        )

        # Define the actor network
        self.actor_net = nn.Sequential(
            self.backbone,
            nn.Linear(64, 2),
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

        self.value_net = nn.Sequential(self.backbone, nn.Linear(64, 1)).to(self.device)

        self.value_module = ValueOperator(module=self.value_net, in_keys=["mask"])
        # self.advantage_module = GAE(
        #     gamma=1, lmbda=0.9, value_network=self.value_module, average_gae=True
        # )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        # You can wrap this in autocast when calling from training loop if needed
        return self.policy_module(td)

    def process_batch(self, td: TensorDictBase):
        """
        Process a batch of data, updating value networks and policy
        """
        # self.advantage_module(td)

        # Zero the gradients
        self.optim.zero_grad()

        # Compute the loss
        loss_td = self.loss_module(td)

        loss: Tensor = loss_td["loss_objective"] + loss_td["loss_critic"]
        if self.entropy_bonus:
            loss += loss_td["loss_entropy"]

        # Backpropagation
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        # Update the policy and value networks
        self.optim.step()

        return float(loss.item()), float(grad_norm.item())

    def save(self, path: str) -> None:
        """
        Save the model weights and hyperparameters to a single file.

        Args:
            path (str): Path to the file where the model and hyperparameters will be saved.
        """
        # Prepare the data to save
        checkpoint = {
            "policy_module_state_dict": self.policy_module.state_dict(),
            "value_module_state_dict": self.value_module.state_dict(),
            "config": {
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
        data = torch.load(path, map_location=device)
        config = data["config"]

        # Create the agent using the loaded hyperparameters
        agent = PPOThresholdAgent(
            device=device,
            lr=config["lr"],
            clip_epsilon=config["clip_epsilon"],
            entropy_bonus=config["entropy_bonus"],
            max_grad_norm=config["max_grad_norm"],
        )

        # Load the model weights
        agent.policy_module.load_state_dict(data["policy_state_dict"])
        agent.value_module.load_state_dict(data["value_state_dict"])

        return agent

    def get_info(self) -> dict:
        info = {
            "value norm": calculate_norm(self.value_net),
            "actor norm": calculate_norm(self.actor_net),
        }
        return info


class DDPGThresholdValueNet(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out_size: int, device: torch.device):
        super().__init__()
        self.backbone = backbone
        assert backbone_out_size // 2 >= 2, "Backbone output size must be at least 4"
        self.backbone_out_size = backbone_out_size
        self.device = device
        self.head = nn.Sequential(
            nn.Linear(backbone_out_size + 1, backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_size // 2, 1),
        ).to(self.device)

    def forward(self, mask: Tensor, threshold: Tensor) -> Tensor:
        # Pass mask through the backbone
        x = self.backbone(mask.to(self.device))  # (N, backbone_out_size)
        # Concatenate the threshold to the output of the backbone
        x = torch.cat((x, threshold.to(self.device)), dim=1)  # (N, backbone_out_size + 1)
        return self.head(x)


class DDPGThresholdAgent(ThresholdAgent):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        update_tau: float,
        replay_buffer_size,
        num_optim: int,
        max_grad_norm=1.0,
    ):
        self.device = device
        self.lr = lr
        self.update_tau = update_tau
        self.replay_buffer_size = replay_buffer_size
        self.num_optim = num_optim
        self.max_grad_norm = max_grad_norm

        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (N, 16, D/2, H/2, W/2)
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (N, 32, D/4, H/4, W/4)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (N, 64, D/8, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),  # -> (N, 64, 1, 1, 1)
            nn.Flatten(start_dim=1),  # -> (N, 64)
        ).to(self.device)

        # Define the actor network
        self.actor_net = nn.Sequential(
            self.backbone,
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.policy_module = TensorDictModule(self.actor_net, in_keys=["mask"], out_keys=["threshold"])

        self.value_net = DDPGThresholdValueNet(
            backbone=self.backbone,
            backbone_out_size=64,
            device=self.device,
        )

        self.value_module = TensorDictModule(
            module=self.value_net, in_keys=["mask", "threshold"], out_keys=["state_action_value"]
        )

        self.loss_module = DDPGLoss(
            actor_network=self.policy_module,
            value_network=self.value_module,
        )
        self.loss_keys = ["loss_objective", "loss_critic"]
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.updater = SoftUpdate(self.loss_module, tau=self.update_tau)

        # Every sample added to the replay buffer (images etc.) can have different shapes, so use a normal list
        self.replay_buffer = TensorDictReplayBuffer(storage=ListStorage(self.replay_buffer_size))

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        return self.policy_module(td.to(self.device))

    def _sample_and_optimize(self):
        """
        Sample a batch from the replay buffer and update network weights
        """
        # Sample a batch from the replay buffer
        td = self.replay_buffer.sample(1)

        # Move the batch to the device
        td = td.to(self.device)

        # Zero the gradients
        self.optim.zero_grad()

        # Compute the loss
        loss_td = self.loss_module(td)

        loss: Tensor = loss_td["loss_actor"] + loss_td["loss_value"]

        # Backpropagation
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        # Update the policy and value networks
        self.optim.step()

        return loss.item(), grad_norm.item()

    def process_batch(self, td: TensorDictBase):
        """
        Process a batch of data, updating value networks and policy
        """

        # Add the batch to the replay buffer
        self.replay_buffer.extend(td.to("cpu"))

        # Sample num_optim times from the replay buffer and optimize
        losses = torch.zeros(self.num_optim, device=self.device)
        grad_norms = torch.zeros(self.num_optim, device=self.device)
        for i in range(self.num_optim):
            loss, grad_norm = self._sample_and_optimize()
            losses[i] = loss
            grad_norms[i] = grad_norm

        # Update the target network
        self.updater.step()

        return float(losses.mean().item()), float(grad_norms.mean().item())

    def save(self, path: Path) -> None:
        """
        Save the model weights and hyperparameters to a single file.

        Args:
            path (str): Path to the file where the model and hyperparameters will be saved.
        """
        # TODO: Save the replay buffer as well
        # Prepare the data to save
        checkpoint = {
            "policy_module_state_dict": self.policy_module.state_dict(),
            "value_module_state_dict": self.value_module.state_dict(),
            "config": {
                "lr": self.lr,
                "update_tau": self.update_tau,
                "replay_buffer_size": self.replay_buffer_size,
                "num_optim": self.num_optim,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        # Save everything in one file
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "DDPGThresholdAgent":
        """
        Load the policy model weights and hyperparameters from a single file.

        Args:
            path (str): Path to the file containing the model and hyperparameters
            device (torch.device): Device to load the model on
        """
        # TODO: Load the replay buffer as well
        # Load the combined checkpoint
        data = torch.load(path, map_location=device)
        config = data["config"]

        # Create the agent using the loaded hyperparameters
        agent = DDPGThresholdAgent(
            device=device,
            lr=config["lr"],
            update_tau=config["update_tau"],
            replay_buffer_size=config["replay_buffer_size"],
            num_optim=config["num_optim"],
            max_grad_norm=config["max_grad_norm"],
        )

        # Load the model weights
        agent.policy_module.load_state_dict(data["policy_module_state_dict"])
        agent.value_module.load_state_dict(data["value_module_state_dict"])

        return agent

    def get_info(self) -> dict:
        info = {
            "value norm": calculate_norm(self.value_net),
            "actor norm": calculate_norm(self.actor_net),
            "replay buffer size": len(self.replay_buffer),
        }
        return info

class AttentionThresholdValueNet(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out_size: int):
        super().__init__()
        self.backbone = backbone
        assert backbone_out_size // 2 >= 2, "Backbone output size must be at least 4"
        self.backbone_out_size = backbone_out_size
        self.head = nn.Sequential(
            nn.Linear(backbone_out_size + 1, backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_size // 2, 1),
        )

    def forward(self, mask: Tensor, threshold: Tensor) -> Tensor:
        # Pass mask through the backbone
        x = self.backbone(mask)  # (N, backbone_out_size)
        # Concatenate the threshold to the output of the backbone
        x = torch.cat((x, threshold), dim=1)  # (N, backbone_out_size + 1)
        return self.head(x)

class AttentionThresholdAgent(ThresholdAgent):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        update_tau: float,
        replay_buffer_size,
        num_optim: int,
        prompt_embedding_dim: int, # Dimensionality of prompt embeddings as produced by the image model
        max_grad_norm=1.0,
    ):
        self.device = device
        self.lr = lr
        self.update_tau = update_tau
        self.replay_buffer_size = replay_buffer_size
        self.num_optim = num_optim
        self.prompt_embedding_dim = prompt_embedding_dim
        self.max_grad_norm = max_grad_norm
        self.backbone_out_size = 64

        self.backbone = PromptAttentionNet(
            num_layers=2,
            emb_dim=self.prompt_embedding_dim,
            num_heads=4,
            output_size=self.backbone_out_size,
        ).to(self.device)

        # Define the actor network
        self.actor_net = nn.Sequential(
            self.backbone,
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 2, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.policy_module = TensorDictModule(self.actor_net, in_keys=["mask"], out_keys=["threshold"])

        self.value_net = AttentionThresholdValueNet(
            backbone=self.backbone,
            backbone_out_size=self.backbone_out_size,
        ).to(self.device)

        self.value_module = TensorDictModule(
            module=self.value_net, in_keys=["mask", "threshold"], out_keys=["state_action_value"]
        )

        self.loss_module = DDPGLoss(
            actor_network=self.policy_module,
            value_network=self.value_module,
        )
        self.loss_keys = ["loss_objective", "loss_critic"]
        self.updater = SoftUpdate(self.loss_module, tau=self.update_tau)
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

        def optim_normalize_hook(optimizer, *args, **kwargs):
            for module in self.backbone.modules():
                if hasattr(module, "normalize_weights"):
                    module.normalize_weights()

        # Every sample added to the replay buffer (images etc.) can have different shapes, so use a normal list
        self.replay_buffer = TensorDictReplayBuffer(storage=ListStorage(self.replay_buffer_size))

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        return self.policy_module(td.to(self.device))

    def _sample_and_optimize(self):
        """
        Sample a batch from the replay buffer and update network weights
        """
        # Sample a batch from the replay buffer
        td = self.replay_buffer.sample(1)

        # Move the batch to the device
        td = td.to(self.device)

        # Zero the gradients
        self.optim.zero_grad()

        # Compute the loss
        loss_td = self.loss_module(td)

        loss: Tensor = loss_td["loss_actor"] + loss_td["loss_value"]

        # Backpropagation
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        # Update the policy and value networks
        self.optim.step()

        return loss.item(), grad_norm.item()

    def process_batch(self, td: TensorDictBase):
        """
        Process a batch of data, updating value networks and policy
        """

        # Add the batch to the replay buffer
        self.replay_buffer.extend(td.to("cpu"))

        # Sample num_optim times from the replay buffer and optimize
        losses = torch.zeros(self.num_optim, device=self.device)
        grad_norms = torch.zeros(self.num_optim, device=self.device)
        for i in range(self.num_optim):
            loss, grad_norm = self._sample_and_optimize()
            losses[i] = loss
            grad_norms[i] = grad_norm

        # Update the target network
        self.updater.step()

        return float(losses.mean().item()), float(grad_norms.mean().item())

    def save(self, path: Path) -> None:
        """
        Save the model weights and hyperparameters to a single file.

        Args:
            path (str): Path to the file where the model and hyperparameters will be saved.
        """
        # TODO: Save the replay buffer as well
        # Prepare the data to save
        checkpoint = {
            "policy_module_state_dict": self.policy_module.state_dict(),
            "value_module_state_dict": self.value_module.state_dict(),
            "config": {
                "lr": self.lr,
                "update_tau": self.update_tau,
                "replay_buffer_size": self.replay_buffer_size,
                "num_optim": self.num_optim,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        # Save everything in one file
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "DDPGThresholdAgent":
        """
        Load the policy model weights and hyperparameters from a single file.

        Args:
            path (str): Path to the file containing the model and hyperparameters
            device (torch.device): Device to load the model on
        """
        # TODO: Load the replay buffer as well
        # Load the combined checkpoint
        data = torch.load(path, map_location=device)
        config = data["config"]

        # Create the agent using the loaded hyperparameters
        agent = DDPGThresholdAgent(
            device=device,
            lr=config["lr"],
            update_tau=config["update_tau"],
            replay_buffer_size=config["replay_buffer_size"],
            num_optim=config["num_optim"],
            max_grad_norm=config["max_grad_norm"],
        )

        # Load the model weights
        agent.policy_module.load_state_dict(data["policy_module_state_dict"])
        agent.value_module.load_state_dict(data["value_module_state_dict"])

        return agent

    def get_info(self) -> dict:
        info = {
            "value norm": calculate_norm(self.value_net),
            "actor norm": calculate_norm(self.actor_net),
            "replay buffer size": len(self.replay_buffer),
        }
        return info



THRESHOLD_AGENT_REGISTRY: dict[str, Type[ThresholdAgent]] = {
    # "PPO": PPOThresholdAgent,
    "DDPG": DDPGThresholdAgent,
    "attention": AttentionThresholdAgent,
}
