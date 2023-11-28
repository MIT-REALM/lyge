import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from model.env.base import ControlAffineSystem


class Agent(ABC):

    def __init__(self, state_dim: int, action_dim: int, gamma: float, device: torch.device):
        super().__init__()
        self.learning_steps = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

    @abstractmethod
    def act(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str, device: torch.device):
        pass

    @abstractmethod
    def set_controller(self, controller: nn.Module):
        pass

    @abstractmethod
    def is_update(self, step: int):
        """
        Whether the time is for update

        Parameters
        ----------
        step: int
            current training step
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the algorithm
        """
        pass

    @abstractmethod
    def step(self, env: ControlAffineSystem, state: torch.Tensor, t: int, step: int):
        """
        Sample one step in the environment
        """
        pass

    def explore(self, x: torch.Tensor, std: float) -> np.array:
        mean = self.act(x)
        return torch.normal(mean=mean, std=std).numpy()
