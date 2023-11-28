import torch
import torch.nn as nn

from typing import Tuple, Callable

from .base import NeuralEnv

from model.network.mlp import NormalizedMLP
from model.env.base import ControlAffineSystem


class NeuralAffineEnv(NeuralEnv):
    """
    dx/dt = f(x) + g(x) u
    """

    def __init__(
            self,
            base_env: ControlAffineSystem,  # to simulate this env
            device: torch.device,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            xdot_mean: torch.Tensor,
            xdot_std: torch.Tensor,
            normalize: bool = False,
            sparse_g: bool = False,
            hidden_layers: tuple = (128, 128, 128),
            hidden_activation: nn.Module = nn.Tanh(),
            goal_relaxation: float = 0.01,
    ):
        super().__init__(
            n_dims=base_env.n_dims,
            n_controls=base_env.n_controls,
            device=device,
            goal_point=base_env.goal_point,
            u_eq=base_env.u_eq,
            xdot_mean=xdot_mean,
            state_std=state_std,
            ctrl_std=ctrl_std,
            xdot_std=xdot_std,
            state_limits=base_env.state_limits,
            control_limits=base_env.control_limits,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            dt=base_env.dt,
            goal_relaxation=goal_relaxation
        )

        self._f_func = NormalizedMLP(
            in_dim=self.n_dims,
            out_dim=self.n_dims,
            input_mean=self.goal_point.squeeze(),
            input_std=self.state_std,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation
        ).to(device)
        self._g_func = NormalizedMLP(
            in_dim=self.n_dims,
            out_dim=self.n_dims * self.n_controls,
            input_mean=self.goal_point.squeeze(),
            input_std=self.state_std,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation
        ).to(device)

        self.base_env = base_env
        self.device = device
        self.normalize = normalize
        self.sparse_g = sparse_g

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u
            dx/dt = f(x, u)

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state
        u: torch.Tensor
            batch_size x self.n_controls tensor of controls

        Returns
        -------
        x_dot: torch.Tensor
            batch_size x self.n_dims tensor of time derivatives of x
        """
        # x_trans = self.normalize_state(x)
        if self.normalize:
            u = self.normalize_action(u)

        # check input
        assert u.ndim == 2 and u.shape[1] == self.n_controls

        # get the control-affine dynamics
        f, g = self.control_affine_dynamics(x)

        # compute state derivatives using control-affine form
        x_dot = f + torch.bmm(g, u.unsqueeze(-1))
        x_dot = x_dot.reshape(x.shape)
        if self.normalize:
            x_dot = self.de_normalize_xdot(x_dot)
        return x_dot

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:
            dx/dt = f(x) + g(x) u

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state
            self.n_dims tensor of state
            batch_size x self.n_dims x 1 tensor of state

        Returns
        -------
        f: torch.Tensor
            batch_size x self.n_dims x 1 representing the control-independent dynamics
        g: torch.Tensor
            batch_size x self.n_dims x self.n_controls representing the control-dependent dynamics
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        return self._f(x), self._g(x)

    # def unnormalized_control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:
    #     if x.ndim == 3:
    #         x_trans = x.squeeze(-1)
    #
    #     x_trans = self.normalize_state(x_trans)
    #
    #     f_bar, g_bar = self.control_affine_dynamics(x_trans)
    #
    #     f = self.xdot_std * (f_bar - torch.matmul(g_bar, (self.u_eq / self.ctrl_std).t())).squeeze(-1) + self.xdot_mean
    #     f = f.unsqueeze(-1)
    #     g = self.xdot_std.repeat(2, 1).t() * g_bar / self.ctrl_std
    #
    #     return f, g

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state

        Returns
        -------
        f: torch.Tensor
            batch_size x self.n_dims x 1
        """
        return self._f_func(x).unsqueeze(-1)

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state

        Returns
        -------
        g: torch.Tensor
            batch_size x self.n_dims x self.n_controls
        """
        g = self._g_func(x)
        if self.sparse_g:
            mask = (torch.abs(g) < 1e-2).float()
            g = g * (1 - mask)
        return g.reshape(x.shape[0], self.n_dims, self.n_controls)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_env.goal_mask(x)

    def sample_states(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_states(batch_size)

    def sample_ctrls(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_ctrls(batch_size)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_goal(batch_size)
