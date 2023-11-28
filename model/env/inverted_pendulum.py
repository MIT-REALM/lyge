import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem, grav


class InvertedPendulum(ControlAffineSystem):
    """
    Inverted pendulum with fixed base

    The system has state
        x[0] = theta (rad)
        x[1] = theta_dot (rad)

    and control inputs
        u = torque (N)

    The system is parameterized by
        m: mass (kg)
        L: length (m)
        b: damping
    """

    # number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # state indices
    THETA = 0
    THETA_DOT = 1

    # control indices
    U = 0

    # max episode steps
    MAX_EPISODE_STEPS = 1000

    # name of the states
    STATE_NAME = [
        'theta',
        'theta_dot'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super(InvertedPendulum, self).__init__(device, dt, params, controller_dt)

    def reset(self) -> torch.Tensor:
        """
        Initialize the state.
            theta: [-0.2, 0.2]
            theta_dot: [-0.2, 0.2]

        Returns
        -------
        state: torch.Tensor
            1 x self.n_dims tensor
        """
        self._state = (torch.rand(1, 2, device=self.device) * 2 - 1) * 0.2
        self._t = 0
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u).type_as(self._state)
        if u.ndim == 1:
            u = u.unsqueeze(0)
        u = u
        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )
        self._state = self.forward(self._state, u)
        self._action = u
        theta = self._state[0, InvertedPendulum.THETA]
        reward = float(2.0 - torch.abs(theta))
        self._t += 1
        done = bool(torch.abs(self.state[InvertedPendulum.THETA]) > 1.5) or self._t >= self.max_episode_steps
        info = dict()
        return self.state, reward, done, info

    def render(self) -> np.ndarray:
        height = 5
        width = 5
        fig = plt.figure(figsize=(height, width), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        theta = self._state[0, InvertedPendulum.THETA].cpu().detach().numpy()
        theta_dot = self._state[0, InvertedPendulum.THETA_DOT].cpu().detach().numpy()
        point0 = self.goal_point.cpu().detach().numpy()
        x = [point0[0, 0]]
        y = [point0[0, 1]]
        x.append(point0[0, 0] + self.params['L'] * np.sin(theta))
        y.append(point0[0, 1] + self.params['L'] * np.cos(theta))
        plt.plot(x, y, linewidth=3)
        plt.xlim([-self.params['L'], self.params['L']])
        plt.ylim([-self.params['L'], self.params['L']])
        text_point = (self.params['L'] - 0.5, self.params['L'] - 0.1)
        line_gap = 0.1
        plt.text(text_point[0], text_point[1], f'T: {self._t}')
        plt.text(text_point[0], text_point[1] - line_gap, f'Theta: {theta:.2f}')
        plt.text(text_point[0], text_point[1] - 2 * line_gap, f'Theta dot: {theta_dot:.2f}')
        if self._action is not None:
            plt.text(text_point[0], text_point[1] - 3 * line_gap,
                     f'U: {self._action.cpu().detach().numpy().item():.2f}')
        ax.axis('off')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        height = 5
        width = 5
        fig = plt.figure(figsize=(height, width), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        theta = state[0, InvertedPendulum.THETA].cpu().detach().numpy()
        theta_dot = state[0, InvertedPendulum.THETA_DOT].cpu().detach().numpy()
        point0 = self.goal_point.cpu().detach().numpy()
        x = [point0[0, 0]]
        y = [point0[0, 1]]
        x.append(point0[0, 0] + self.params['L'] * np.sin(theta))
        y.append(point0[0, 1] + self.params['L'] * np.cos(theta))
        plt.plot(x, y, linewidth=3)
        plt.xlim([-self.params['L'], self.params['L']])
        plt.ylim([-self.params['L'], self.params['L']])
        text_point = (self.params['L'] - 0.5, self.params['L'] - 0.1)
        line_gap = 0.1
        plt.text(text_point[0], text_point[1], f'T: {t}')
        plt.text(text_point[0], text_point[1] - line_gap, f'Theta: {theta:.2f}')
        plt.text(text_point[0], text_point[1] - 2 * line_gap, f'Theta dot: {theta_dot:.2f}')
        if action is not None:
            plt.text(text_point[0], text_point[1] - 3 * line_gap, f'U: {action.cpu().detach().numpy().item():.2f}')
        ax.axis('off')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    def default_param(self) -> dict:
        return {"m": 1.0, "L": 1.0, "b": 0.01}

    def validate_params(self, params: dict) -> bool:
        """
        Check if a given set of parameters is valid.

        Parameters
        ----------
        params: dict
            parameter values for the system. Requires keys ["m", "L", "b"]

        Returns
        -------
        valid: bool
            True if parameters are valid, False otherwise
        """
        valid = True

        # make sure all needed parameters were provided
        valid = valid and "m" in params
        valid = valid and "L" in params
        valid = valid and "b" in params

        # make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["L"] > 0
        valid = valid and params["b"] > 0

        return valid

    def state_name(self, dim: int) -> str:
        return InvertedPendulum.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        return float(torch.norm(state))

    @property
    def n_dims(self) -> int:
        return InvertedPendulum.N_DIMS

    @property
    def n_controls(self) -> int:
        return InvertedPendulum.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return InvertedPendulum.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_dims, device=self.device)
        upper_limit[InvertedPendulum.THETA] = 2.0
        upper_limit[InvertedPendulum.THETA_DOT] = 2.0

        lower_limit = -1.0 * upper_limit

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([100.], device=self.device)
        lower_limit = -torch.tensor([100.], device=self.device)

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        # extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # extract the needed parameters and state variables
        m, L, b = self.params["m"], self.params["L"], self.params["b"]
        theta = x[:, InvertedPendulum.THETA]
        theta_dot = x[:, InvertedPendulum.THETA_DOT]

        # the derivatives of theta is just its velocity
        f[:, InvertedPendulum.THETA, 0] = theta_dot

        # acceleration in theta depends on theta via gravity and theta_dot via damping
        f[:, InvertedPendulum.THETA_DOT, 0] = (
                grav / L * theta - b / (m * L ** 2) * theta_dot
        )

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        # extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # extract the needed parameters
        m, L = self.params["m"], self.params["L"]

        # effect on theta dot
        g[:, InvertedPendulum.THETA_DOT, InvertedPendulum.U] = 1 / (m * L ** 2)

        return g

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(x.shape[0]).type_as(x)
        dist = torch.norm((x - self.goal_point), dim=1)
        for i in range(dist.shape[0]):
            if dist[i] > 0.01:
                mask[i] = 0
        return mask.bool()

