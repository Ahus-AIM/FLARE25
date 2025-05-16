from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from src.rl.agents import DDPGAgent, serializable
from src.rl.models import PromptAttentionNet


class AttentionDDPGThresholdValueNet(nn.Module):
    def __init__(self, backbone_out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(backbone_out_size + 1, backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_size // 2, backbone_out_size // 4),
            nn.ReLU(),
            nn.Linear(backbone_out_size // 4, 1),
        )

    def forward(self, backbone_out, threshold):
        # Concatenate the backbone output and threshold
        x = torch.cat((backbone_out, threshold), dim=-1)
        return self.net(x)


@dataclass(kw_only=True, eq=False, order=False)
class AttentionDDPGThresholdAgent(DDPGAgent):
    prompt_embedding_dim: int = serializable()
    backbone_out_size: int = serializable()

    def pre_init_hook(self) -> None:
        self.backbone = TensorDictModule(
            PromptAttentionNet(
                num_layers=2,
                emb_dim=self.prompt_embedding_dim,
                num_heads=4,
                output_size=self.backbone_out_size,
            ),
            in_keys=["padded_prompt_embeddings", "prompt_embedding_attention_mask"],
            out_keys=["backbone_out"],
        )

    def post_init_hook(self) -> None:
        def optim_normalize_hook(optimizer, *args, **kwargs):
            for module in self.backbone.modules():
                if hasattr(module, "normalize_weights"):
                    module.normalize_weights()

        self.optimizer.register_step_post_hook(optim_normalize_hook)

    def get_state_action_value_module(self) -> TensorDictModule:
        state_action_value_net = AttentionDDPGThresholdValueNet(self.backbone_out_size)

        state_action_value_module = TensorDictSequential(
            [
                self.backbone,
                TensorDictModule(
                    state_action_value_net,
                    in_keys=["backbone_out", "threshold"],
                    out_keys=["state_action_value"],
                ),
            ]
        )
        return state_action_value_module

    def get_policy_module(self) -> TensorDictModule:
        actor_net = nn.Sequential(
            nn.Linear(self.backbone_out_size, self.backbone_out_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 2, self.backbone_out_size // 4),
            nn.ReLU(),
            nn.Linear(self.backbone_out_size // 4, 1),
            nn.Sigmoid(),
        )

        deterministic_policy_module = TensorDictSequential(
            [
                self.backbone,
                TensorDictModule(actor_net, in_keys=["backbone_out"], out_keys=["threshold"]),
            ]
        )
        return deterministic_policy_module
