import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Bounded
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal


class PPOThresholdAgent:
    def __init__(self, device: str):
        self.device = torch.device(device)
        actor_net = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(3, 3, 3)),  # (N, 1, 3, 3, 3)
            # flatten last 4 dimensions
            nn.Flatten(start_dim=1),  # (N, 1*3*3*3)
            nn.Linear(1 * 3 * 3 * 3, 2, dtype=torch.float16),  # (N, 2)
            NormalParamExtractor(),
        ).to(device)
        policy_module = TensorDictModule(actor_net, in_keys=["mask"], out_keys=["loc", "scale"])
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=Bounded(low=0, high=1, shape=torch.Size(()), dtype=torch.float16),
            in_keys=["loc", "scale"],
            out_keys=["threshold"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": 0, "high": 1},
            return_log_prob=True,
        )

    def policy(self, td: TensorDictBase) -> TensorDictBase:
        self.policy_module(td)
        return td
