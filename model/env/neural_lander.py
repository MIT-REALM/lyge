import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem
from model.network.fa_network import FaNetwork


class NeuralLander(ControlAffineSystem):
    """
    Represents a neural lander (a 3D quadrotor with learned ground effect).
    """

    rho = 1.225
    gravity = 9.81
    drone_height = 0.09
    mass = 1.47

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 3

    # State indices
    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5

    # Control indices
    AX = 0
    AY = 1
    AZ = 2

    # max episode steps
    MAX_EPISODE_STEPS = 1000

    # name of the states
    STATE_NAME = [
        'PX',
        'PY',
        'PZ',
        'VX',
        'VY',
        'VZ'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None,
    ):
        super(NeuralLander, self).__init__(device, dt, params, controller_dt)

        self.Fa_model = FaNetwork().to(self.device)
        self.Fa_model.disable_grad()
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.Fa_model.load_state_dict(
            torch.load(dir_name + '/data/Fa_net_12_3_full_Lip16.pth',
                       map_location=self.device)
        )

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._state = torch.zeros(1, self.n_dims, device=self.device)
        self._state[0, NeuralLander.PX].copy_(torch.rand(1).squeeze(0) * 4 - 2)
        self._state[0, NeuralLander.PY].copy_(torch.rand(1).squeeze(0) * 4 - 2)
        self._state[0, NeuralLander.PZ].copy_(torch.rand(1).squeeze(0) + 1)
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
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

        # calculate returns
        self._state = self.forward(self._state, u)
        self._action = u
        self._t += 1
        upper_x_lim, lower_x_limit = self.state_limits
        done = self._t >= self.max_episode_steps or \
               (self._state[0, 0:3] > upper_x_lim[:3]).any() or (self._state[0, :3] < lower_x_limit[:3]).any()
        reward = float(10.0 - torch.norm(self._state[3:]) - torch.norm(self._state[:3] - self.goal_point))

        return self.state, reward, done, {}

    def render(self) -> np.ndarray:
        # plot background
        h = 500
        w = 500
        fig = plt.figure(figsize=(h / 100, w / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.goal_point[0, 0], self.goal_point[0, 1], self.goal_point[0, 2], s=80)

        # plot drone
        state = self.state
        x_drone = state[NeuralLander.PX].cpu().detach().numpy()
        y_drone = state[NeuralLander.PY].cpu().detach().numpy()
        z_drone = state[NeuralLander.PZ].cpu().detach().numpy()
        vx_drone = state[NeuralLander.VX].cpu().detach().numpy()
        vy_drone = state[NeuralLander.VY].cpu().detach().numpy()
        vz_drone = state[NeuralLander.VZ].cpu().detach().numpy()
        ax.scatter3D(x_drone, y_drone, z_drone, s=100)

        # set limits
        upper_limit, lower_limit = self.state_limits
        upper_limit = upper_limit.cpu().numpy()
        lower_limit = lower_limit.cpu().numpy()
        ax.set_xlim((lower_limit[NeuralLander.PX], upper_limit[NeuralLander.PX]))
        ax.set_ylim((lower_limit[NeuralLander.PY], upper_limit[NeuralLander.PY]))
        ax.set_zlim((lower_limit[NeuralLander.PZ], upper_limit[NeuralLander.PZ]))

        # text
        text_point = (0, 1.1)
        line_gap = 0.04
        ax.text2D(text_point[0], text_point[1], f'T: {self._t}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - line_gap, f'PX: {x_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 2 * line_gap, f'VX: {vx_drone:.2f}', transform=ax.transAxes)
        if self._action is not None:
            ax.text2D(text_point[0], text_point[1] - 3 * line_gap,
                      f'AX: {self._action[0, NeuralLander.AX]:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 4 * line_gap, f'PY: {y_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 5 * line_gap, f'VY: {vy_drone:.2f}', transform=ax.transAxes)
        if self._action is not None:
            ax.text2D(text_point[0], text_point[1] - 6 * line_gap,
                      f'AY: {self._action[0, NeuralLander.AY]:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 7 * line_gap, f'PZ: {z_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 8 * line_gap, f'VZ: {vz_drone:.2f}', transform=ax.transAxes)
        if self._action is not None:
            ax.text2D(text_point[0], text_point[1] - 9 * line_gap,
                      f'AZ: {self._action[0, NeuralLander.AZ]:.2f}', transform=ax.transAxes)

        # get rgb array
        ax.axis('on')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        # plot background
        h = 500
        w = 500
        fig = plt.figure(figsize=(h / 100, w / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.goal_point[0, 0], self.goal_point[0, 1], self.goal_point[0, 2], s=80)

        # plot drone
        x_drone = state[0, NeuralLander.PX].cpu().detach().numpy()
        y_drone = state[0, NeuralLander.PY].cpu().detach().numpy()
        z_drone = state[0, NeuralLander.PZ].cpu().detach().numpy()
        vx_drone = state[0, NeuralLander.VX].cpu().detach().numpy()
        vy_drone = state[0, NeuralLander.VY].cpu().detach().numpy()
        vz_drone = state[0, NeuralLander.VZ].cpu().detach().numpy()
        ax.scatter3D(x_drone, y_drone, z_drone, s=100)

        # set limits
        upper_limit, lower_limit = self.state_limits
        upper_limit = upper_limit.cpu().numpy()
        lower_limit = lower_limit.cpu().numpy()
        ax.set_xlim((lower_limit[NeuralLander.PX], upper_limit[NeuralLander.PX]))
        ax.set_ylim((lower_limit[NeuralLander.PY], upper_limit[NeuralLander.PY]))
        ax.set_zlim((lower_limit[NeuralLander.PZ], upper_limit[NeuralLander.PZ]))

        # text
        text_point = (0, 1.1)
        line_gap = 0.04
        ax.text2D(text_point[0], text_point[1], f'T: {t}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - line_gap, f'PX: {x_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 2 * line_gap, f'VX: {vx_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 3 * line_gap,
                  f'AX: {action[0, NeuralLander.AX]:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 4 * line_gap, f'PY: {y_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 5 * line_gap, f'VY: {vy_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 6 * line_gap,
                  f'AY: {action[0, NeuralLander.AY]:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 7 * line_gap, f'PZ: {z_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 8 * line_gap, f'VZ: {vz_drone:.2f}', transform=ax.transAxes)
        ax.text2D(text_point[0], text_point[1] - 9 * line_gap,
                  f'AZ: {action[0, NeuralLander.AZ]:.2f}', transform=ax.transAxes)

        # get rgb array
        ax.axis('on')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        return {}

    def validate_params(self, params: dict) -> bool:
        return True

    def state_name(self, dim: int) -> str:
        return NeuralLander.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        return float(torch.norm(state[:3] - self.goal_point[:, :3]))

    @property
    def n_dims(self) -> int:
        return NeuralLander.N_DIMS

    @property
    def n_controls(self) -> int:
        return NeuralLander.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return NeuralLander.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_dims, device=self.device)
        upper_limit[NeuralLander.PX] = 8.0
        upper_limit[NeuralLander.PY] = 8.0
        upper_limit[NeuralLander.PZ] = 5.0
        upper_limit[NeuralLander.VX] = 1.0
        upper_limit[NeuralLander.VY] = 1.0
        upper_limit[NeuralLander.VZ] = 1.0

        lower_limit = -1.0 * upper_limit
        lower_limit[NeuralLander.PZ] = -0.5

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([50, 50, 50], device=self.device)
        lower_limit = -1.0 * upper_limit

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Derivatives of positions are just velocities
        f[:, NeuralLander.PX, 0] = x[:, NeuralLander.VX]  # x
        f[:, NeuralLander.PY, 0] = x[:, NeuralLander.VY]  # y
        f[:, NeuralLander.PZ, 0] = x[:, NeuralLander.VZ]  # z

        # Constant acceleration in z due to gravity
        f[:, NeuralLander.VZ, 0] = -NeuralLander.gravity

        # Add disturbance from ground effect
        _, _, z, vx, vy, vz = [x[:, i] for i in range(self.n_dims)]
        Fa = self.Fa_func(z, vx, vy, vz) / NeuralLander.mass
        f[:, NeuralLander.VX, 0] += Fa[:, 0]
        f[:, NeuralLander.VY, 0] += Fa[:, 1]
        f[:, NeuralLander.VZ, 0] += Fa[:, 2]

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Linear accelerations are control variables
        g[:, NeuralLander.VX:, :] = torch.eye(self.n_controls) / NeuralLander.mass

        return g

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = x.norm(dim=-1) <= 0.3
        goal_mask = torch.logical_and(goal_mask, near_goal)

        return goal_mask

    def Fa_func(self, z, vx, vy, vz):
        if next(self.Fa_model.parameters()).device != z.device:
            self.Fa_model.to(z.device)

        bs = z.shape[0]

        # use prediction from NN as ground truth
        state = torch.zeros([bs, 1, 12]).type_as(z)
        state[:, 0, 0] = z + NeuralLander.drone_height
        state[:, 0, 1] = vx  # velocity
        state[:, 0, 2] = vy  # velocity
        state[:, 0, 3] = vz  # velocity
        state[:, 0, 7] = 1.0
        state[:, 0, 8:12] = 6508.0 / 8000
        state = state.float()

        Fa = self.Fa_model(state).squeeze(1) * torch.tensor([30.0, 15.0, 10.0], device=self.device).reshape(
            1, 3
        ).type_as(z)
        return Fa.type_as(z)

    def compute_A_matrix(self) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        A = np.zeros((self.n_dims, self.n_dims))
        A[:NeuralLander.PZ + 1, NeuralLander.VX:] = np.eye(3)

        return A

    @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls), device=self.device)
        u_eq[0, NeuralLander.AZ] = NeuralLander.mass * NeuralLander.gravity
        return u_eq

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        LQR controller behaves too badly in this environment, so we change to use PID controller.
        """
        coef = torch.tensor([8., 8., 8., 1., 1., 1.]).type_as(x) * NeuralLander.mass
        goal = self.goal_point.squeeze().type_as(x)
        error = -coef * (x - goal)
        if error.ndim == 1:
            error = error.unsqueeze(0)
            u_nominal = torch.cat([
                error[:, 0] + error[:, 3],
                error[:, 1] + error[:, 4],
                error[:, 2] + error[:, 5]
            ], dim=0).unsqueeze(0)
        else:
            u_nominal = torch.cat([
                (error[:, 0] + error[:, 3]).unsqueeze(1),
                (error[:, 1] + error[:, 4]).unsqueeze(1),
                (error[:, 2] + error[:, 5]).unsqueeze(1)
            ], dim=1)
        return u_nominal + self.u_eq.type_as(x)

    @property
    def goal_point(self) -> torch.Tensor:
        goal = torch.zeros((1, self.n_dims), device=self.device)
        goal[0, 2] = 0.5
        return goal
