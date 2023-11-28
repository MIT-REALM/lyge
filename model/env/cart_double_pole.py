import gym
import torch
import numpy as np

from typing import Optional, Tuple
from gym.spaces import Box

from .base import ControlAffineSystem


class CartDoublePole(ControlAffineSystem):

    # name of the states
    STATE_NAME = [
        'x',
        # 'sin(theta_1)',
        # 'sin(theta_2)',
        # 'cos(theta_1)',
        # 'cos(theta_2)',
        'theta_1',
        'theta_2',
        'v',
        'theta_dot_1',
        'theta_dot_2'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.1,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super(CartDoublePole, self).__init__(device, dt, params, controller_dt)
        self.base = gym.make('InvertedDoublePendulum-v2')

    def reset(self) -> torch.Tensor:
        state = self.base.reset()
        self._state = torch.tensor([state[0], np.arcsin(state[1]), np.arcsin(state[2]),
                                    state[5], state[6], state[7]], device=self.device, dtype=torch.float)
        # self._state = torch.from_numpy(state).type(torch.float).to(self.device)[:8]
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
        self._state = torch.tensor([next_state[0], np.arcsin(next_state[1]), np.arcsin(next_state[2]),
                                    next_state[5], next_state[6], next_state[7]], device=self.device, dtype=torch.float)
        # self._state = torch.from_numpy(next_state).type_as(self.state)[:8]
        self._action = u
        self._t += 1
        reward -= abs(float(self.state[0]))
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
        return CartDoublePole.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        return float(torch.norm(state[0]))

    @property
    def n_dims(self) -> int:
        return 6

    @property
    def n_controls(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 1000

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
        return Box(low=-np.inf, high=np.inf, shape=(self.n_dims,), dtype=np.float32)

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
            qpos = x[i, :3]
            qvel = x[i, 3:]
            self.base.reset()
            self.base.set_state(qpos, qvel)
            ob, _, _, _ = self.base.step(u[i])
            x_next[i, :] = np.array([ob[0], np.arcsin(ob[1]), np.arcsin(ob[2]), ob[5], ob[6], ob[7]])
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
        # return torch.tensor([[0., 0., 0., 0., 0., 0.]], device=self.device)

    @property
    def reward_range(self) -> tuple:
        return self.base.reward_range

    @property
    def metadata(self) -> dict:
        return self.base.metadata

    @property
    def use_lqr(self) -> bool:
        return False
