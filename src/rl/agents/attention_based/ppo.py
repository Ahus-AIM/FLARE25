
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor, nn
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss

from src.rl.agents import ThresholdAgent
from src.rl.models import PromptAttentionNet
from src.rl.utils import calculate_norm


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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = (x * mask.unsqueeze(-1)).nan_to_num(0.0)  # (batch_size, n_points, prompt_embedding_dim)
        x = self.net1(x)  # (1, n_points, output_size)
        x = x.permute(0, 2, 1)  # (1, output_size, n_points)
        x = F.avg_pool1d(x, kernel_size=x.shape[-1])  # (1, output_size, 1)
        x = x.squeeze(-1)  # (1, output_size)
        return x


# class AttentionPPOThresholdPolicyNet(nn.Module):
#     def __init__(self, backbone: PromptAttentionNet, backbone_out_size: int):
#         super().__init__()
#         self.backbone = backbone
#         self.backbone_out_size = backbone_out_size
#         self.head = nn.Sequential(
#             nn.Linear(backbone_out_size, backbone_out_size // 2),
#             nn.ReLU(),
#             nn.Linear(backbone_out_size // 2, 2),
#             NormalParamExtractor(),
#         )

#     def forward(self, padded_prompt_embeddings: PromptEmbeddings, prompt_embedding_attention_mask: Tensor) -> Tensor:
#         x = self.backbone(padded_prompt_embeddings, prompt_embedding_attention_mask)
#         return self.head(x)

# class AttentionPPOThresholdValueNet(nn.Module):
#     def __init__(self, backbone: PromptAttentionNet, backbone_out_size: int):
#         super().__init__()
#         self.backbone = backbone
#         self.backbone_out_size = backbone_out_size
#         self.head = nn.Sequential(
#             nn.Linear(backbone_out_size, backbone_out_size // 2),
#             nn.ReLU(),
#             nn.Linear(backbone_out_size // 2, 1),
#         )

#     def forward(self, padded_prompt_embeddings: PromptEmbeddings, prompt_embedding_attention_mask: Tensor) -> Tensor:
#         x = self.backbone(padded_prompt_embeddings, prompt_embedding_attention_mask)
#         return self.head(x)


class AttentionPPOThresholdAgent(ThresholdAgent):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        num_optim: int,
        prompt_embedding_dim: int,  # Dimensionality of prompt embeddings as produced by the image model
        max_grad_norm=1.0,
    ):
        self.device = device
        self.lr = lr
        self.num_optim = num_optim
        self.prompt_embedding_dim = prompt_embedding_dim
        self.max_grad_norm = max_grad_norm
        self.backbone_out_size = 64

        # self.backbone = PromptAttentionNet(
        #     num_layers=2,
        #     emb_dim=self.prompt_embedding_dim,
        #     num_heads=4,
        #     output_size=self.backbone_out_size,
        # ).to(self.device)

        self.backbone = TensorDictModule(
            PromptAttentionNet(
                num_layers=2,
                emb_dim=self.prompt_embedding_dim,
                num_heads=4,
                output_size=self.backbone_out_size,
            ),
            in_keys=["padded_prompt_embeddings", "prompt_embedding_attention_mask"],
            out_keys=["backbone_out"],
        ).to(self.device)

        # Define the actor network
        self.actor_net = nn.Sequential(
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 2, self.backbone_out_size // 4),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 4, 2),
            NormalParamExtractor(),
        ).to(self.device)

        self.deterministic_policy_module = TensorDictSequential(
            [self.backbone, TensorDictModule(self.actor_net, in_keys=["backbone_out"], out_keys=["loc", "scale"])]
        )

        self.probabilistic_policy_module = ProbabilisticActor(
            module=self.deterministic_policy_module,
            in_keys=["loc", "scale"],
            out_keys=["threshold"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": 0, "high": 1},
            return_log_prob=True,
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 2, self.backbone_out_size // 4),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 4, 1),
        ).to(self.device)

        self.value_module = TensorDictSequential(
            [
                self.backbone,
                TensorDictModule(module=self.value_net, in_keys=["backbone_out"], out_keys=["state_value"]),
            ]
        )

        self.loss_module = ClipPPOLoss(
            self.probabilistic_policy_module,
            self.value_module,
        )

        self.loss_keys = ["loss_critic", "loss_entropy", "loss_objective"]
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

        def optim_normalize_hook(optimizer, *args, **kwargs):
            for module in self.backbone.modules():
                if hasattr(module, "normalize_weights"):
                    module.normalize_weights()

        self.optim.register_step_post_hook(optim_normalize_hook)

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        return self.probabilistic_policy_module(td.to(self.device)).to(td.device)

    def process_batch(self, td: TensorDictBase):
        # Move the batch to the device
        td = td.to(self.device)

        # Zero the gradients
        self.optim.zero_grad()

        # Compute the loss
        loss_td = self.loss_module(td)

        loss: Tensor = sum(loss_td[k] for k in self.loss_keys)  # type: noqa

        # Backpropagation
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        # Update the policy and value networks
        self.optim.step()

        return loss.item(), grad_norm.item()

    def get_info(self) -> dict:
        info = {
            "value norm": calculate_norm(self.value_net),
            "actor norm": calculate_norm(self.actor_net),
            # "replay buffer size": len(self.replay_buffer),
        }
        return info
