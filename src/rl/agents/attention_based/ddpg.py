from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.data import ListStorage, PrioritizedSampler, TensorDictReplayBuffer
from torchrl.objectives import DDPGLoss, SoftUpdate

from src.custom_types import PromptEmbeddings, Threshold
from src.rl.agents import ThresholdAgent
from src.rl.agents.mask_based.ddpg import DDPGThresholdAgent
from src.rl.models import PromptAttentionNet
from src.rl.utils import calculate_norm


class AttentionThresholdValueNet(nn.Module):
    def __init__(self, backbone: PromptAttentionNet, backbone_out_size: int):
        super().__init__()
        self.backbone = backbone
        assert backbone_out_size // 2 >= 2, "Backbone output size must be at least 4"
        self.backbone_out_size = backbone_out_size
        self.head = nn.Sequential(
            nn.Linear(backbone_out_size + 1, backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_size // 2, 1),
        )

    def forward(self, prompt_embeddings: PromptEmbeddings, threshold: Threshold) -> Tensor:
        # Pass prompt embeddings through the backbone
        x = self.backbone(prompt_embeddings)  # (N, backbone_out_size)
        # Concatenate the threshold to the output of the backbone
        x = torch.cat((x, threshold), dim=1)  # (N, backbone_out_size + 1)
        return self.head(x)


class DummyBackbone(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.net1 = nn.Sequential(
            # input is (batch_size, n_points, prompt_embedding_dim)
            nn.Linear(self.input_size, self.input_size // 2),
            nn.ReLU(),
            nn.Linear(self.input_size // 2, self.input_size // 4),
            nn.ReLU(),
            nn.Linear(self.input_size // 4, self.output_size),  # (batch_size, n_points, output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net1(x)  # (batch_size, n_points, output_size)
        x = x.permute(0, 2, 1)  # (batch_size, output_size, n_points)
        x = F.avg_pool1d(x, kernel_size=x.shape[-1])  # (batch_size, output_size, 1)
        x = x.squeeze(-1)  # (batch_size, output_size)
        return x


class AttentionDDPGThresholdAgent(ThresholdAgent):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        update_tau: float,
        replay_buffer_size,
        num_optim: int,
        prompt_embedding_dim: int,  # Dimensionality of prompt embeddings as produced by the image model
        max_grad_norm=1.0,
        rb_alpha: float = 0.6,
        rb_beta: float = 0.4,
        # limits for the threshold
        lowest_threshold: float = 0.0,
        highest_threshold: float = 1.0,
    ):
        self.device = device
        self.lr = lr
        self.update_tau = update_tau
        self.replay_buffer_size = replay_buffer_size
        self.num_optim = num_optim
        self.prompt_embedding_dim = prompt_embedding_dim
        self.max_grad_norm = max_grad_norm
        self.rb_alpha = rb_alpha
        self.rb_beta = rb_beta
        self.backbone_out_size = 64

        # self.backbone = PromptAttentionNet(
        #     num_layers=2,
        #     emb_dim=self.prompt_embedding_dim,
        #     num_heads=4,
        #     output_size=self.backbone_out_size,
        # ).to(self.device)

        self.backbone = DummyBackbone(
            input_size=self.prompt_embedding_dim,
            output_size=self.backbone_out_size,
        ).to(self.device)

        # Define the actor network
        self.actor_net = nn.Sequential(
            self.backbone,
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.Tanh(),
            nn.Linear(self.backbone_out_size // 2, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.policy_module = TensorDictModule(self.actor_net, in_keys=["prompt_embeddings"], out_keys=["threshold"])

        self.value_net = AttentionThresholdValueNet(
            backbone=self.backbone,
            backbone_out_size=self.backbone_out_size,
        ).to(self.device)

        self.value_module = TensorDictModule(
            module=self.value_net, in_keys=["prompt_embeddings", "threshold"], out_keys=["state_action_value"]
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

        self.optim.register_step_post_hook(optim_normalize_hook)

        # Every sample added to the replay buffer (images etc.) can have different shapes, so use a normal list
        self.replay_buffer = TensorDictReplayBuffer(
            storage=ListStorage(self.replay_buffer_size),
            sampler=PrioritizedSampler(max_capacity=self.replay_buffer_size, alpha=self.rb_alpha, beta=self.rb_beta),
        )

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        return self.policy_module(td.to(self.device)).to(td.device)

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

        # Update the priorities in the replay buffer
        self.replay_buffer.update_tensordict_priority(td)

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
