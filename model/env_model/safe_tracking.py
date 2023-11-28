import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

from .affine import NeuralAffineEnv

from model.env.base import TrackingEnv
from model.buffer import TrackingBuffer
from model.network.mlp import MLP


class NeuralSafeCriticalTrackingEnv(NeuralAffineEnv):

    def __init__(
            self,
            base_env: TrackingEnv,  # to simulate this env
            device: torch.device,
            safe_buffer: TrackingBuffer,
            g_prior: bool = False,
            sparse_g: bool = True,
            hidden_layers: tuple = (128, 128),
            hidden_activation: nn.Module = nn.Tanh(),
            goal_relaxation: float = 0.1
    ):
        states, actions, next_states = safe_buffer.all_transitions
        xdots = (next_states - states) / base_env.dt

        super().__init__(
            base_env=base_env,
            device=device,
            xdot_mean=torch.mean(xdots, dim=0),
            state_std=torch.std(states, dim=0),
            ctrl_std=torch.std(actions, dim=0),
            xdot_std=torch.std(xdots, dim=0),
            sparse_g=sparse_g,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            goal_relaxation=goal_relaxation,
        )

        self.safe_buffer = safe_buffer

        if g_prior:
            self._g_func = MLP(
                in_dim=self.n_dims,
                out_dim=self.n_controls * self.n_controls,
                hidden_layers=hidden_layers,
                hidden_activation=hidden_activation
            ).to(device)
        self.g_prior = g_prior

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        if not self.g_prior:
            g = self._g_func(x).reshape(x.shape[0], self.n_dims, self.n_controls)
        else:
            g1 = self._g_func(x).reshape(x.shape[0], self.n_controls, self.n_controls)
            g2 = torch.zeros(x.shape[0], self.n_dims - self.n_controls, self.n_controls).type_as(x)
            g = torch.cat((g2, g1), dim=1)
        if self.sparse_g:
            mask = (torch.abs(g) < 1e-2).float()
            g = g * (1 - mask)
        return g

    def sample_safe_states(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_states(batch_size)

    def sample_safe_states_ref(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.safe_buffer.sample_states_ref(batch_size)

    def sample_states_ref(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_env.sample_states_ref(batch_size)

    def sample_safe_ctrls(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_policy(batch_size)[1]
