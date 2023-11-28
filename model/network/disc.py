import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import NormalizedMLP


class AIRLDiscrim(nn.Module):
    """
    Discriminator used by AIRL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations
    """
    def __init__(
            self,
            state_dim: int,
            goal_point: torch.Tensor,
            state_std: torch.Tensor,
            gamma: float,
            hidden_units_r: tuple = (128, 128),
            hidden_units_v: tuple = (128, 128),
            hidden_activation_r: nn.Module = nn.ReLU(inplace=True),
            hidden_activation_v: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.g = NormalizedMLP(
            in_dim=state_dim,
            out_dim=1,
            input_mean=goal_point.squeeze(),
            input_std=state_std,
            hidden_layers=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = NormalizedMLP(
            in_dim=state_dim,
            out_dim=1,
            input_mean=goal_point.squeeze(),
            input_std=state_std,
            hidden_layers=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate the f(s, s') function
        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        next_states: torch.Tensor
            next state corresponding to the current state
        Returns
        -------
        f: value of the f(s, s') function
        """
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Output the discriminator's result sigmoid(f - log_pi) without sigmoid
        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state
        Returns
        -------
        result: f - log_pi
        """
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reward using AIRL's learned reward signal f
        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state
        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            # return logits
            return -F.logsigmoid(-logits)

    def get_reward(self, states: torch.Tensor) -> torch.Tensor:
        return self.g(states)
