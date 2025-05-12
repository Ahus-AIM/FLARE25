from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, Type, TypeVar

import torch
import yaml
from tensordict import TensorDictBase
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleBase,
)
from torch import Tensor, nn, optim
from torchrl.data import (
    LazyTensorStorage,
    ListStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.objectives.value import GAE

T = TypeVar("T", bound="Agent")


def field_with_metadata(field_type: str, default=MISSING, default_factory=MISSING, init: bool = True):
    """Create a dataclass field with metadata."""
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    metadata = {"type": field_type}
    # If default is provided, prefer it over default_factory
    if default is not MISSING:
        return field(default=default, init=init, metadata=metadata)
    elif default_factory is not MISSING:
        return field(default_factory=default_factory, init=init, metadata=metadata)
    else:
        return field(init=init, metadata=metadata)


def serializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as serializable (saved to params.yml)."""
    return field_with_metadata("serializable", default, default_factory, init)


def weights(default=MISSING, default_factory=MISSING, init: bool = False):
    """Mark a field as containing model weights (saved to model.pt)."""
    return field_with_metadata("weights", default, default_factory, init)


def unserializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as unserializable (saved to extra.pt)."""
    return field_with_metadata("unserializable", default, default_factory, init)


class Agent(ABC):
    # Subclasses will be dataclasses
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @abstractmethod
    def process_batch(self, td: TensorDictBase) -> dict[str, float]: ...

    def get_train_info(self) -> dict[str, Any]:
        return {}

    def get_eval_info(self) -> dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def policy(self) -> TensorDictModuleBase: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @device.setter
    @abstractmethod
    def device(self, device: torch.device) -> None:
        """Move agent networks to a specific device."""
        ...

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save the class name
        with open(path / "class_name.txt", "w") as f:
            f.write(self.__class__.__name__)

        # Save weights
        weights_dict = {
            field.name: getattr(self, field.name).state_dict()
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "weights"
        }
        torch.save(weights_dict, path / "model.pt")

        # Save unserializable fields
        unserializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "unserializable"
        }
        torch.save(unserializable_dict, path / "extra.pt")

        # Save serializable fields
        serializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "serializable"
        }
        with open(path / "params.yml", "w") as f:
            yaml.dump(serializable_dict, f)

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        # Load the class name
        with open(path / "class_name.txt") as f:
            class_name = f.read().strip()

        # Find the subclass, even if it's several layers down
        def find_subclass(base_class: type, name: str) -> type | None:
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == name:
                    return subclass
                found = find_subclass(subclass, name)
                if found:
                    return found
            return None

        subclass = find_subclass(cls, class_name)
        if subclass is None:
            raise ValueError(f"Unknown subclass: {class_name}")

        # Load serializable fields
        with open(path / "params.yml") as f:
            serializable_dict = yaml.safe_load(f)

        # Load unserializable fields
        unserializable_dict = torch.load(path / "extra.pt", weights_only=False)

        # Merge all fields
        full_dict = {**serializable_dict, **unserializable_dict}

        # Construct agent
        agent = subclass(**full_dict)

        # Load weights
        weights_dict = torch.load(path / "model.pt")
        for _field in agent.__dataclass_fields__.values():
            if _field.metadata.get("type") == "weights":
                getattr(agent, _field.name).load_state_dict(weights_dict[_field.name])

        return agent


@dataclass(kw_only=True, eq=False, order=False)
class PPOAgent(Agent, ABC):
    """Proximal Policy Optimization (PPO) agent."""

    action_spec: TensorSpec = unserializable(default=None)

    # Device. All modules created by subclasses will be moved to this device.
    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # PPO parameters
    gamma: float = unserializable(default=1)
    lmbda: float = unserializable(default=0.95)
    clip_epsilon: float = unserializable(default=0.2)  # weight clipping threshold
    entropy_bonus: bool = unserializable(default=True)  # whether to encourage exploration
    entropy_coef: float = unserializable(default=1e-4)  # how much to weight the entropy loss term
    critic_coef: float = unserializable(default=1.0)  # how much to weight the critic loss term
    loss_critic_type: str = unserializable(default="smooth_l1")  # what type of loss to use for the critic

    # Optimizer parameters
    lr: float = unserializable(default=1e-3)
    max_grad_norm: float = unserializable(default=1)

    # Replay buffer parameters
    batch_size: int = unserializable(default=1000)
    sub_batch_size: int = unserializable(default=100)  # size of batch when sampling from replay buffer
    num_epochs: int = unserializable(default=10)
    replay_buffer_device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # Set in constructor
    policy_module: ProbabilisticTensorDictSequential = weights(init=False)
    state_value_module: TensorDictModule = weights(init=False)
    advantage_module: GAE = field(init=False)
    loss_module: ClipPPOLoss = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def pre_init_hook(self) -> None:
        """Hook for subclasses to optionally run code before __post_init__."""
        pass

    def post_init_hook(self) -> None:
        """Hook for subclasses to optionally run code after __post_init__."""
        pass

    @abstractmethod
    def get_policy_module(self) -> ProbabilisticTensorDictSequential:
        """Get the policy module."""
        pass

    @abstractmethod
    def get_state_value_module(self) -> TensorDictModule:
        """Get the state value module."""
        pass

    def __post_init__(self) -> None:
        # Ensure batch_size is divisible by sub_batch_size
        if self.batch_size % self.sub_batch_size != 0:
            raise ValueError("batch_size must be divisible by sub_batch_size.")

        self.pre_init_hook()

        self.policy_module = self.get_policy_module().to(self._device)
        self.state_value_module = self.get_state_value_module().to(self._device)

        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.state_value_module,
            average_gae=self.sub_batch_size > 1,  # we cannot average or calculate std with a single sample
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.state_value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
        )
        self.loss_keys = ["loss_objective", "loss_critic"] + (["loss_entropy"] if self.entropy_bonus else [])
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        # Since every element will have different size, we have to use a ListStorage
        # self.replay_buffer = TensorDictReplayBuffer(
        #     storage=ListStorage(max_size=self.batch_size),
        #     sampler=SamplerWithoutReplacement(),
        #     batch_size=self.sub_batch_size,
        # )

        self.post_init_hook()

    @property
    def policy(self) -> ProbabilisticTensorDictSequential:
        return self.policy_module

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        assert td.batch_size == torch.Size((self.batch_size,)), "Batch size mismatch"

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}

        # Perform multiple epochs of training
        for _ in range(self.num_epochs):
            # Compute advantages each epoch
            self.advantage_module(td)

            # Reset replay buffer each epoch
            # self.replay_buffer.extend(td)  # type: ignore

            # loss_td = self._sample_and_train()
            self.optimizer.zero_grad()
            loss_td: TensorDictBase = self.loss_module(td)
            loss_tensor: Tensor = sum((loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device))
            loss_tensor.backward()
            nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            for k in self.loss_keys:
                total_loss_td[k] += loss_td[k].item()

        # Compute average loss
        num_updates = self.num_epochs * (self.batch_size // self.sub_batch_size)
        process_td = {k: v / num_updates for k, v in total_loss_td.items()}
        process_td["advantage"] = td["advantage"].mean().item()
        process_td["value_target"] = td["value_target"].mean().item()

        return process_td

    # def _sample_and_train(self) -> TensorDictBase:
    #     """Sample from the replay buffer and train the policy, returning the loss td."""
    #     td = self.replay_buffer.sample()
    #     td = td.to(self._device)

    #     self.optimizer.zero_grad()
    #     loss_td: TensorDictBase = self.loss_module(td)
    #     loss_tensor: Tensor = sum((loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device))
    #     loss_tensor.backward()
    #     nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)
    #     self.optimizer.step()

    #     return loss_td

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.state_value_module = self.state_value_module.to(self._device)


@dataclass(kw_only=True, eq=False, order=False)
class DDPGAgent(Agent, ABC):
    """Deep Deterministic Policy Gradient (DDPG) agent."""

    action_spec: TensorSpec = unserializable()

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # DDPG parameters
    gamma: float = serializable(default=0.99)
    loss_function: str = serializable(default="l2")
    delay_actor: bool = serializable(default=False)
    delay_value: bool = serializable(default=True)
    separate_losses: bool = serializable(default=False)

    # target network update rate
    update_tau: float = serializable(default=0.005)

    # Optimizer parameters
    lr: float = serializable(default=1e-3)
    max_grad_norm: float = serializable(default=1)

    # Replay buffer parameters
    replay_buffer_size: int = serializable(default=1000)
    sub_batch_size: int = serializable(default=100)
    num_samples: int = serializable(default=10)
    replay_buffer_alpha: float = serializable(default=0.6)
    replay_buffer_beta_init: float = serializable(default=0.4)
    replay_buffer_beta_end: float = serializable(default=1.0)
    replay_buffer_beta_annealing_num_batches: int = serializable(default=10000)
    init_random_frames: int = serializable(default=0)
    replay_buffer_device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # Set in constructor
    policy_module: TensorDictModule = weights(init=False)
    state_action_value_module: TensorDictModule = weights(init=False)
    loss_module: DDPGLoss = field(init=False)
    target_net_updater: SoftUpdate = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def pre_init_hook(self) -> None:
        """Hook for subclasses to optionally run code before __post_init__."""
        pass

    def post_init_hook(self) -> None:
        """Hook for subclasses to optionally run code after __post_init__."""
        pass

    @abstractmethod
    def get_policy_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    @abstractmethod
    def get_state_action_value_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    def __post_init__(self) -> None:
        self.pre_init_hook()

        self.policy_module = self.get_policy_module().to(self._device)
        self.state_action_value_module = self.get_state_action_value_module().to(self._device)

        self.loss_module = DDPGLoss(
            actor_network=self.policy_module,
            value_network=self.state_action_value_module,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - self.update_tau)
        self.loss_keys = ["loss_actor", "loss_value"]
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.replay_buffer_size, device=self.replay_buffer_device),
            sampler=PrioritizedSampler(
                max_capacity=self.replay_buffer_size,
                alpha=self.replay_buffer_alpha,
                beta=self.replay_buffer_beta_init,
            ),
            priority_key="td_error",
            batch_size=self.sub_batch_size,
        )

        self.post_init_hook()

    @property
    def policy(self) -> TensorDictModule:
        return self.policy_module

    def _anneal_replay_buffer_beta(self) -> None:
        """Anneal the beta parameter for prioritized sampling."""
        if self.replay_buffer.sampler.beta < self.replay_buffer_beta_end:
            self.replay_buffer.sampler.beta += (
                self.replay_buffer_beta_end - self.replay_buffer_beta_init
            ) / self.replay_buffer_beta_annealing_num_batches

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        # Add to replay buffer
        self.replay_buffer.extend(td)  # type: ignore

        # Only continue if we have enough samples in the replay buffer, in which case we return 0.0 for all losses
        if len(self.replay_buffer) < self.init_random_frames:
            return {k: 0.0 for k in self.loss_keys}

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}

        for _ in range(self.num_samples):
            loss_td = self._sample_and_train()

            # Accumulate losses
            for k in self.loss_keys:
                total_loss_td[k] += loss_td[k].item()

        # Update target network
        self.target_net_updater.step()

        # Anneal beta for prioritized sampling
        self._anneal_replay_buffer_beta()

        # Compute average loss
        avg_loss_td = {k: v / self.num_samples for k, v in total_loss_td.items()}

        return avg_loss_td

    def _sample_and_train(self) -> TensorDictBase:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self.device)

        self.optimizer.zero_grad()
        loss_td: TensorDictBase = self.loss_module(td)
        loss_tensor: Tensor = sum((loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device))
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_tensordict_priority(td)  # type: ignore

        return loss_td

    def get_train_info(self) -> dict[str, Any]:
        """Get training information."""
        return {
            "replay_buffer_beta": self.replay_buffer.sampler.beta,
            "replay_buffer_size": len(self.replay_buffer),
        }

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.action_value_module = self.action_value_module.to(self._device)
