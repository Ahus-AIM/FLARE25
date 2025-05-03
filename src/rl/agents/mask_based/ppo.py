import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import Tensor, nn
from torchrl.data import Bounded
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss

from src.rl.agents import ThresholdAgent
from src.rl.utils import calculate_norm


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
