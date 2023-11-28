import torch
import torch.nn as nn

from .base import NeuralEnv

from model.env.base import ControlAffineSystem
from model.buffer import DemoBuffer


class NeuralSafeCriticalEnv(NeuralEnv):

    def __init__(
            self,
            base_env: ControlAffineSystem,  # to simulate this env
            device: torch.device,
            safe_buffer: DemoBuffer,
            normalize: bool = False,
            sparse_g: bool = False,
            hidden_layers: tuple = (128, 128, 128),
            hidden_activation: nn.Module = nn.Tanh(),
            goal_relaxation: float = 0.1
    ):
        states, actions, next_states = safe_buffer.all_transitions
        xdots = (next_states - states) / base_env.dt

        super(NeuralSafeCriticalEnv, self).__init__(
            n_dims=base_env.n_dims,
            n_controls=base_env.n_controls,
            device=device,
            goal_point=base_env.goal_point,
            u_eq=base_env.u_eq,
            xdot_mean=torch.mean(xdots, dim=0),
            # state_std=torch.ones(base_env.n_dims, device=device),
            # ctrl_std=torch.ones(base_env.n_controls, device=device),
            # xdot_std=torch.ones(base_env.n_dims, device=device),
            state_std=torch.std(states, dim=0),
            ctrl_std=torch.std(actions, dim=0),
            xdot_std=torch.std(xdots, dim=0),
            state_limits=base_env.state_limits,
            control_limits=base_env.control_limits,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            dt=base_env.dt,
            goal_relaxation=goal_relaxation
        )

        self.base_env = base_env
        self.normalize = normalize
        self.device = device
        self.safe_buffer = safe_buffer

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            u = self.normalize_action(u)

        x_dot = self._f_func(torch.cat((x, u), dim=1))

        if self.normalize:
            x_dot = self.de_normalize_xdot(x_dot)
        return x_dot

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_env.goal_mask(x)

    def sample_states(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_states(batch_size)

    def sample_ctrls(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_ctrls(batch_size)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        return self.base_env.sample_goal(batch_size)

    def sample_safe_states(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_states(batch_size)

    def sample_safe_ctrls(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_policy(batch_size)[1]

