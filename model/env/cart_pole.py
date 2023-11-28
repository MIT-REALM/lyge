import gym
import torch
import numpy as np

from typing import Optional
from gym.spaces import Box

from .base import ControlAffineSystem, Tuple


class CartPole(ControlAffineSystem):
    # number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # state indices
    X = 0
    THETA = 1
    V = 2
    THETA_DOT = 3

    # control indices
    U = 0

    # max episode steps
    MAX_EPISODE_STEPS = 1000

    # name of the states
    STATE_NAME = [
        'x',
        'theta',
        'v',
        'theta_dot'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.1,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super(CartPole, self).__init__(device, dt, params, controller_dt)
        self.base = gym.make('InvertedPendulum-v2')

    def reset(self) -> torch.Tensor:
        state = self.base.reset()
        self._state = torch.from_numpy(state).type(torch.float).to(self.device)
        self._t = 0
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u).type_as(self._state)
        if u.ndim == 1:
            u = u.unsqueeze(0)
        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )
        next_state, reward, done, info = self.base.step(u.squeeze(0).cpu().detach().numpy())
        reward -= float(torch.norm(self.state))
        self._state = torch.from_numpy(next_state).type_as(self.state)
        self._action = u
        self._t += 1
        return self.state, reward, done, info

    def render(self, mode='rgb_array') -> np.ndarray:
        return self.base.render(mode)

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        raise NotImplementedError

    def default_param(self) -> dict:
        return {}

    def validate_params(self, params: dict) -> bool:
        return True

    def state_name(self, dim: int) -> str:
        return CartPole.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        return float(torch.norm(state[0]))

    @property
    def n_dims(self) -> int:
        return CartPole.N_DIMS

    @property
    def n_controls(self) -> int:
        return CartPole.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return CartPole.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.base.observation_space, Box)
        upper_limit = torch.from_numpy(self.base.observation_space.high).type(torch.float).to(self.device)
        lower_limit = torch.from_numpy(self.base.observation_space.low).type(torch.float).to(self.device)
        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.base.action_space, Box)
        upper_limit = torch.from_numpy(self.base.action_space.high).type(torch.float).to(self.device)
        lower_limit = torch.from_numpy(self.base.action_space.low).type(torch.float).to(self.device)
        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(x.shape[0]).type_as(x)
        dist = torch.norm((x - self.goal_point), dim=1)
        for i in range(dist.shape[0]):
            if dist[i] > 0.01:
                mask[i] = 0
        return mask.bool()

    @property
    def observation_space(self) -> Box:
        assert isinstance(self.base.observation_space, Box)
        return self.base.observation_space

    @property
    def action_space(self) -> Box:
        assert isinstance(self.base.action_space, Box)
        return self.base.action_space

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # DO NOT call this during simulation!
        x_cp = x
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if u.ndim == 1:
            u = u.unsqueeze(0)
        x = x.detach().cpu()
        u = u.detach().cpu()
        x_next = np.empty((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            qpos = x[i, :2]
            qvel = x[i, 2:]
            self.base.reset()
            self.base.set_state(qpos, qvel)
            ob, _, _, _ = self.base.step(u[i])
            x_next[i, :] = ob
        return torch.from_numpy(x_next).type_as(x_cp)

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_A_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def compute_B_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def linearized_ct_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def linearized_dt_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_linearized_controller(self):
        raise NotImplementedError

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_states(self, batch_size: int) -> torch.Tensor:
        result = torch.empty(batch_size, self.n_dims).to(self.device)
        for i in range(batch_size):
            result[i, :] = torch.from_numpy(self.observation_space.sample()).type(torch.float).to(self.device)
        return result

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def sample_states_ref(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def noisy_u_nominal(self, x: torch.Tensor, std: float, bias: float = 0) -> torch.Tensor:
        raise NotImplementedError

    @property
    def goal_point(self) -> torch.Tensor:
        return torch.zeros((1, self.n_dims), device=self.device)

    @property
    def reward_range(self) -> tuple:
        return self.base.reward_range

    @property
    def metadata(self) -> dict:
        return self.base.metadata

    @property
    def use_lqr(self) -> bool:
        return False
