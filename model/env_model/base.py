import torch
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

from model.network.mlp import NormalizedMLP


class NeuralEnv(nn.Module, ABC):
    """
    Discrete environment defined by neural networks
    """

    def __init__(
            self,
            n_dims: int,
            n_controls: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            xdot_mean: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            xdot_std: torch.Tensor,
            state_limits: Tuple[torch.Tensor, torch.Tensor],
            control_limits: Tuple[torch.Tensor, torch.Tensor],
            hidden_layers: tuple = (128, 128, 128),
            hidden_activation: nn.Module = nn.Tanh(),
            dt: float = 0.01,
            goal_relaxation: float = 0.01
    ):
        super().__init__()

        self._n_dims = n_dims
        self._n_controls = n_controls
        self._device = device
        self._goal_point = goal_point
        self.u_eq = u_eq
        self.xdot_mean = xdot_mean
        self.state_std = state_std
        self.ctrl_std = ctrl_std
        self.xdot_std = xdot_std
        self._state_limits = state_limits
        self._control_limits = control_limits
        self._dt = dt
        self._goal_relaxation = goal_relaxation

        self._f_func = NormalizedMLP(
            in_dim=self.n_dims + self.n_controls,
            out_dim=self.n_dims,
            input_mean=torch.cat((goal_point.squeeze(0), u_eq.squeeze(0)), dim=0),
            input_std=torch.cat((state_std, ctrl_std), dim=0),
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            limit_lip=False
        ).to(device)

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.state_std)
        zero_mask = torch.ones(self.state_std.shape[0]).type_as(self.state_std)
        zero_mask[nonzero_std_dim] = 0
        x_trans = (x - self.goal_point) / (self.state_std + zero_mask)
        return x_trans

    def normalize_action(self, u: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.ctrl_std)
        zero_mask = torch.ones(self.ctrl_std.shape[0]).type_as(self.ctrl_std)
        zero_mask[nonzero_std_dim] = 0
        u_trans = (u - self.u_eq) / (self.ctrl_std + zero_mask)
        return u_trans

    def normalize_xdot(self, xdot: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.xdot_std)
        zero_mask = torch.ones(self.xdot_std.shape[0]).type_as(self.xdot_std)
        zero_mask[nonzero_std_dim] = 0
        xdot_trans = (xdot - self.xdot_mean) / (self.xdot_std + zero_mask)
        return xdot_trans

    def de_normalize_xdot(self, xdot_trans: torch.Tensor) -> torch.Tensor:
        return xdot_trans * self.xdot_std + self.xdot_mean

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
        # u_trans = self.normalize_action(u)
        # xdot_trans = self._f_func(torch.cat((x, u), dim=1))
        # return self.de_normalize_xdot(xdot_trans)
        return self._f_func(torch.cat((x, u), dim=1))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Simulate one step, return x_{t+1} = x_t + x_dot * dt
        """
        x_dot = self.closed_loop_dynamics(x, u)
        return x + (x_dot * self._dt)

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def goal_point(self) -> torch.Tensor:
        return self._goal_point

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        return self._state_limits

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        return self._control_limits

    @property
    def dt(self) -> float:
        return self._dt

    def sample_states(self, batch_size: int) -> torch.Tensor:
        """
        sample states from the env

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        states: torch.Tensor
            sampled states from the env
        """
        high, low = self.state_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            states = torch.randn(batch_size, self.n_dims, device=self._device)  # todo: add mean and std
        else:
            rand = torch.rand(batch_size, self.n_dims, device=self._device)
            states = rand * (high - low) + low
        return states

    def sample_ctrls(self, batch_size: int) -> torch.Tensor:
        high, low = self.control_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            ctrls = torch.randn(batch_size, self.n_dims, device=self.device)  # todo: add mean and std
        else:
            rand = torch.rand(batch_size, self.n_dims, device=self.device)
            ctrls = rand * (high - low) + low
        return ctrls

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        rand = 2 * torch.rand(batch_size, self.n_dims, device=self.device) - 1
        goal_points = self._goal_point + self._goal_relaxation * rand
        return goal_points

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        judge x is in the goal region or not
        Parameters
        ----------
        x: torch.Tensor
            input states

        Returns
        -------
        mask: torch.Tensor
            same size as x, 1 means in the goal region and 0 means not
        """
        mask = torch.ones(x.shape[0]).type_as(x)
        x_trans = self.normalize_state(x)
        dist = torch.norm(x_trans, dim=1)
        for i in range(dist.shape[0]):
            if dist[i] > self._goal_relaxation:
                mask[i] = 0
        return mask.bool()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self._device))
