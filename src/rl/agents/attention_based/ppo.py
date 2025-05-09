from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import Tensor, nn
from torchrl.modules import (
    ActorValueOperator,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
)

from src.rl.agents import PPOAgent, serializable
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


@dataclass(kw_only=True, eq=False, order=False)
class AttentionPPOThresholdAgent(PPOAgent):
    prompt_embedding_dim: int = serializable()
    backbone_out_size: int = serializable()

    def pre_init_hook(self) -> None:
        # self.backbone_net = DummyBackbone(
        #     input_size=self.prompt_embedding_dim,
        #     output_size=self.backbone_out_size,
        # )
        self.backbone_net = PromptAttentionNet(
            num_layers=2,
            emb_dim=self.prompt_embedding_dim,
            num_heads=4,
            output_size=self.backbone_out_size,
        )

        self.backbone = TensorDictModule(
            self.backbone_net,
            # PromptAttentionNet(
            #     num_layers=2,
            #     emb_dim=self.prompt_embedding_dim,
            #     num_heads=4,
            #     output_size=self.backbone_out_size,
            # ),
            in_keys=["padded_prompt_embeddings", "prompt_embedding_attention_mask"],
            out_keys=["backbone_out"],
        )

        self.actor_net = nn.Sequential(
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 2, self.backbone_out_size // 4),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 4, 2),
            NormalParamExtractor(),
        )

        self.actor_head = ProbabilisticActor(
            module=TensorDictModule(self.actor_net, in_keys=["backbone_out"], out_keys=["loc", "scale"]),
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
        )

        self.value_head = TensorDictModule(module=self.value_net, in_keys=["backbone_out"], out_keys=["state_value"])

        self.actor_value = ActorValueOperator(self.backbone, self.actor_head, self.value_head)

    def post_init_hook(self) -> None:

        def optim_normalize_hook(optimizer, *args, **kwargs):
            for module in self.backbone.modules():
                if hasattr(module, "normalize_weights"):
                    module.normalize_weights()

        self.optimizer.register_step_post_hook(optim_normalize_hook)

    def get_policy_module(self) -> ProbabilisticTensorDictSequential:
        # Define the actor network

        return self.actor_value.get_policy_operator()

    def get_state_value_module(self) -> TensorDictModule:
        return self.actor_value.get_value_operator()

    def get_eval_info(self) -> dict[str, Any]:
        return {
            "actor net norm": calculate_norm(self.actor_net),
            "value net norm": calculate_norm(self.value_net),
            "backbone net norm": calculate_norm(self.backbone_net),
        }
