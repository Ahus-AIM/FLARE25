from pathlib import Path
from typing import Protocol, Type

import torch
from tensordict import TensorDictBase


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
