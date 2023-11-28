import torch
import torch.nn as nn

from typing import Tuple

from .mlp import NormalizedMLP


class TwinnedStateActionFunction(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_point: torch.Tensor,
        u_eq: torch.Tensor,
        state_std: torch.Tensor,
        ctrl_std: torch.Tensor,
        hidden_units: tuple = (128, 128),
        hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net1 = NormalizedMLP(
            in_dim=state_dim + action_dim,
            out_dim=1,
            input_mean=torch.cat((goal_point.squeeze(0), u_eq.squeeze(0)), dim=0),
            input_std=torch.cat((state_std, ctrl_std), dim=0),
            hidden_layers=hidden_units,
            hidden_activation=hidden_activation
        )

        self.net2 = NormalizedMLP(
            in_dim=state_dim + action_dim,
            out_dim=1,
            input_mean=torch.cat((goal_point.squeeze(0), u_eq.squeeze(0)), dim=0),
            input_std=torch.cat((state_std, ctrl_std), dim=0),
            hidden_layers=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net1(torch.cat([states, actions], dim=-1))
