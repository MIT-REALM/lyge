import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from scipy.io import loadmat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import rad2deg, deg2rad
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem
from .utils import scale3d, rotate3d

import model.env.aerobench as aerobench_loader
from aerobench.highlevel.controlled_f16 import controlled_f16  # type: ignore
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot  # type: ignore
from aerobench.lowlevel.low_level_controller import LowLevelController  # type: ignore
from aerobench.visualize.anim3d import get_script_path  # type: ignore
from aerobench.visualize import plot  # type: ignore


class F16Tracking(ControlAffineSystem):
    """
    F16 Tracking environment.

    The system has state
        x[0] = air speed, VT    (ft/sec)
        x[1] = angle of attack, alpha  (rad)
        x[2] = angle of sideslip, beta (rad)
        x[3] = roll angle, phi  (rad)
        x[4] = pitch angle, theta  (rad)
        x[5] = yaw angle, psi  (rad)
        x[6] = roll rate, P  (rad/sec)
        x[7] = pitch rate, Q  (rad/sec)
        x[8] = yaw rate, R  (rad/sec)
        x[9] = northward horizontal displacement, pn  (feet)
        x[10] = eastward horizontal displacement, pe  (feet)
        x[11] = altitude, h  (feet)
        x[12] = engine thrust dynamics lag state, pow
        x[13, 14, 15] = internal integrator states

    and control inputs, which are setpoints for a lower-level integrator
        u[0] = Z acceleration
        u[1] = stability roll rate
        u[2] = throttle command (0.0, 1.0)

    The system is parameterized by
        lag_error: the additive error in the engine lag state dynamics
    """

    # number of states and controls
    N_DIMS = 16
    N_CONTROLS = 3

    # state indices
    VT = 0  # airspeed
    ALPHA = 1  # angle of attack
    BETA = 2  # sideslip angle
    PHI = 3  # roll angle
    THETA = 4  # pitch angle
    PSI = 5  # yaw angle
    Proll = 6  # roll rate
    Q = 7  # pitch rate
    R = 8  # yaw rate
    POSN = 9  # northward displacement
    POSE = 10  # eastward displacement
    H = 11  # altitude
    POW = 12  # engine thrust dynamics lag state

    # control indices
    U_NZ = 0  # desired z acceleration
    U_SR = 1  # desired stability roll rate
    U_THROTTLE = 2  # throttle command

    # max episode steps
    MAX_EPISODE_STEPS = 500

    # stable level
    STABLE_LEVEL = 1000

    # name of the states
    STATE_NAME = [
        'airspeed',
        'angle of attack',
        'sideslip angle',
        'roll angle',
        'pitch angle',
        'yaw angle',
        'roll rate',
        'pitch rate',
        'yaw rate',
        'northward displacement',
        'eastward displacement',
        'altitude',
        'engine thrust dynamics lag',
        'integral 1',
        'integral 2',
        'integral 3'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.05,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None,
    ):
        super(F16Tracking, self).__init__(device, dt, params, controller_dt)

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        self.P = torch.eye(self.n_dims)

        # tracking point: [POSE, POSN, ALT]
        self._setpoint = torch.tensor([5000., 7500., 1500.], dtype=torch.float, device=self.device)
        self._reach = False

    def reset(self) -> torch.Tensor:
        initial_conditions = torch.tensor([
            (520.0, 560.0),  # vt
            (deg2rad(2.1215), deg2rad(2.1215)),  # alpha
            (-0.0, 0.0),  # beta
            (-0.1, 0.1),  # phi
            (-0.1, 0.1),  # theta
            (-0.1, 0.1),  # psi
            (-0.5, 0.5),  # P
            (-0.5, 0.5),  # Q
            (-0.5, 0.5),  # R
            (0.0, 0.0),  # PN
            (0.0, 0.0),  # PE
            (1500.0, 1500.0),  # H
            (4.0, 5.0),  # pow
            (0.0, 0.0),  # integrator state 1
            (0.0, 0.0),  # integrator state 2
            (0.0, 0.0),  # integrator state 3
        ], dtype=torch.float, device=self.device)
        self._t = 0
        self._reach = False
        self._state = torch.rand(1, self.n_dims, device=self.device)
        self._state = self._state * (initial_conditions[:, 1] - initial_conditions[:, 0]) + initial_conditions[:, 0]
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
        done = False
        if torch.isinf(self.state).any() or torch.isnan(self.state).any():
            done = True
        else:
            next_state = self.forward(self._state, u)
            if torch.isinf(next_state).any() or torch.isnan(next_state).any():
                done = True
            else:
                self._state = next_state
        self._action = u
        self._t += 1
        done = self._t >= self.max_episode_steps or done
        upper_x_lim, lower_x_lim = self.state_limits
        done = (self.state < lower_x_lim).any() or (self.state > upper_x_lim).any() or done
        position = self.state[[F16Tracking.POSE, F16Tracking.POSN, F16Tracking.H]]
        reward = float(- torch.norm(position - self._setpoint) * 0.0001)
        # if done:
        #     # if (self.state < lower_x_lim).any():
        #     #     violation = torch.nonzero((self.state < lower_x_lim).detach()).squeeze(-1)
        #     #     for v in violation:
        #     #         tqdm.write(f'F16 error: violate lower bound: {self.state_name(v)}, '
        #     #                    f'value: {self.state[v]:.2f}, time: {self._t}')
        #     # if (self.state > upper_x_lim).any():
        #     #     violation = torch.nonzero((self.state > upper_x_lim).detach()).squeeze(-1)
        #     #     for v in violation:
        #     #         tqdm.write(f'F16 error: violate upper bound: {self.state_name(v)}, '
        #     #                    f'value: {self.state[v]:.2f}, time: {self._t}')
        #     reward -= 1000
        reach = torch.norm(position - self._setpoint) < 400
        if reach and not self._reach:
            reward += 1000
            self._reach = True
        if torch.norm(position - self._setpoint) < 400:
            done = True

        return self.state, reward, done, {}

    def render(self) -> np.ndarray:
        plot.init_plot()
        fig = plt.figure(figsize=(8, 7), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')

        parent = plot.get_script_path()
        plane_point_data = os.path.join(parent, 'f-16.mat')
        data = loadmat(plane_point_data)
        f16_pts = data['V']
        f16_faces = data['F']
        plane_polys = Poly3DCollection([], color='k')
        ax.add_collection3d(plane_polys)
        elev = 30
        azim = 45
        ax.view_init(elev, azim)

        state = self.state.cpu().detach().numpy()
        dx = state[F16Tracking.POSE]
        dy = state[F16Tracking.POSN]
        dz = state[F16Tracking.H]
        pos = np.array([dx, dy, dz])
        distance = np.linalg.norm(pos - self._setpoint.cpu().detach().numpy())

        ax.set_xlabel('X [ft]', fontsize=14)
        ax.set_ylabel('Y [ft]', fontsize=14)
        ax.set_zlabel('Altitude [ft]', fontsize=14)

        # text
        fontsize = 14
        time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
        x_text = ax.text2D(0.5, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')
        y_text = ax.text2D(0.95, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
        dis_text = ax.text2D(0.5, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')
        v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
        beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

        time_text.set_text(f't = {self._t}')
        x_text.set_text(f'x = {state[F16Tracking.POSE]:.2f} ft')
        y_text.set_text(f'y = {state[F16Tracking.POSN]:.2f} ft')
        alt_text.set_text(f'h = {state[F16Tracking.H]:.2f} ft')
        dis_text.set_text(f'dis = {distance:.2f} ft')
        v_text.set_text(f'V = {state[F16Tracking.VT]:.2f} ft/sec')
        alpha_text.set_text(f'$\\alpha$ = {rad2deg(state[F16Tracking.ALPHA]):.2f} deg')
        beta_text.set_text(f'$\\beta$ = {rad2deg(state[F16Tracking.BETA]):.2f} deg')
        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(
            rad2deg(state[F16Tracking.PHI]), rad2deg(state[F16Tracking.THETA]), rad2deg(state[F16Tracking.PSI])))

        # set the space for F16
        s = 30
        pts = scale3d(f16_pts, [-s, s, s])
        pts = rotate3d(pts, state[F16Tracking.THETA], state[F16Tracking.PSI] - math.pi / 2, -state[F16Tracking.PHI])
        size = 1000
        ax.set_xlim([dx - size, dx + size])
        ax.set_ylim([dy - size, dy + size])
        ax.set_zlim([dz - size, dz + size])

        verts = []
        fc = []
        ec = []

        # draw ground
        if dz - size <= 0 <= dz + size:
            z = 0
            verts.append([(dx - size, dy - size, z),
                          (dx + size, dy - size, z),
                          (dx + size, dy + size, z),
                          (dx - size, dy + size, z)])
            fc.append('0.8')
            ec.append('0.8')

        # draw f16
        for face in f16_faces:
            face_pts = []

            for findex in face:
                face_pts.append((pts[findex - 1][0] + dx,
                                 pts[findex - 1][1] + dy,
                                 pts[findex - 1][2] + dz))

            verts.append(face_pts)
            fc.append('0.2')
            ec.append('0.2')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolor(fc)
        plane_polys.set_edgecolor(ec)

        # plot waypoint
        waypoint = self._setpoint.cpu().detach().numpy()
        ax.plot(waypoint[0], waypoint[1], waypoint[2], 'bo', ms=8, lw=0, zorder=50)

        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        plot.init_plot()
        fig = plt.figure(figsize=(8, 7), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')

        parent = plot.get_script_path()
        plane_point_data = os.path.join(parent, 'f-16.mat')
        data = loadmat(plane_point_data)
        f16_pts = data['V']
        f16_faces = data['F']
        plane_polys = Poly3DCollection([], color='k')
        ax.add_collection3d(plane_polys)
        elev = 30
        azim = 45
        ax.view_init(elev, azim)

        state = state.squeeze(0).cpu().detach().numpy()
        dx = state[F16Tracking.POSE]
        dy = state[F16Tracking.POSN]
        dz = state[F16Tracking.H]
        pos = np.array([dx, dy, dz])
        distance = np.linalg.norm(pos - self._setpoint.cpu().detach().numpy())

        ax.set_xlabel('X [ft]', fontsize=14)
        ax.set_ylabel('Y [ft]', fontsize=14)
        ax.set_zlabel('Altitude [ft]', fontsize=14)

        # text
        fontsize = 14
        time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
        x_text = ax.text2D(0.5, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')
        y_text = ax.text2D(0.95, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
        dis_text = ax.text2D(0.5, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')
        v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
        beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

        time_text.set_text(f't = {t}')
        x_text.set_text(f'x = {state[F16Tracking.POSE]:.2f} ft')
        y_text.set_text(f'y = {state[F16Tracking.POSN]:.2f} ft')
        alt_text.set_text(f'h = {state[F16Tracking.H]:.2f} ft')
        dis_text.set_text(f'dis = {distance:.2f} ft')
        v_text.set_text(f'V = {state[F16Tracking.VT]:.2f} ft/sec')
        alpha_text.set_text(f'$\\alpha$ = {rad2deg(state[F16Tracking.ALPHA]):.2f} deg')
        beta_text.set_text(f'$\\beta$ = {rad2deg(state[F16Tracking.BETA]):.2f} deg')
        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(
            rad2deg(state[F16Tracking.PHI]), rad2deg(state[F16Tracking.THETA]), rad2deg(state[F16Tracking.PSI])))

        # set the space for F16
        s = 30
        pts = scale3d(f16_pts, [-s, s, s])
        pts = rotate3d(pts, state[F16Tracking.THETA], state[F16Tracking.PSI] - math.pi / 2, -state[F16Tracking.PHI])
        size = 1000
        ax.set_xlim([dx - size, dx + size])
        ax.set_ylim([dy - size, dy + size])
        ax.set_zlim([dz - size, dz + size])

        verts = []
        fc = []
        ec = []

        # draw ground
        if dz - size <= 0 <= dz + size:
            z = 0
            verts.append([(dx - size, dy - size, z),
                          (dx + size, dy - size, z),
                          (dx + size, dy + size, z),
                          (dx - size, dy + size, z)])
            fc.append('0.8')
            ec.append('0.8')

        # draw f16
        for face in f16_faces:
            face_pts = []

            for findex in face:
                face_pts.append((pts[findex - 1][0] + dx,
                                 pts[findex - 1][1] + dy,
                                 pts[findex - 1][2] + dz))

            verts.append(face_pts)
            fc.append('0.2')
            ec.append('0.2')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolor(fc)
        plane_polys.set_edgecolor(ec)

        # plot waypoint
        waypoint = self._setpoint.cpu().detach().numpy()
        ax.plot(waypoint[0], waypoint[1], waypoint[2], 'bo', ms=8, lw=0, zorder=50)

        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        return {'lag_error': 0.0}

    def validate_params(self, params: dict) -> bool:
        valid = 'lag_error' in params

        return valid

    def state_name(self, dim: int) -> str:
        return F16Tracking.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        pos = torch.tensor([state[10], state[9], state[11]])
        return float(torch.norm(pos - self._setpoint))

    @property
    def n_dims(self) -> int:
        return F16Tracking.N_DIMS

    @property
    def n_controls(self) -> int:
        return F16Tracking.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return F16Tracking.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_limit = torch.tensor(
            [
                300,  # vt
                -1.0,  # alpha
                -1.0,  # beta
                -np.pi,  # phi
                -np.pi,  # theta
                -np.pi,  # psi
                -np.pi,  # P
                -np.pi,  # Q
                -np.pi,  # R
                -2000,  # pos_n
                -2000,  # pos_e
                300.0,  # alt
                0.0,  # pow
                -5.0,  # nz_int
                -5.0,  # ps_int
                -5.0,  # nyr_int
            ], device=self.device
        )
        upper_limit = torch.tensor(
            [
                1000,  # vt
                1.0,  # alpha
                1.0,  # beta
                np.pi,  # phi
                np.pi,  # theta
                np.pi,  # psi
                np.pi,  # P
                np.pi,  # Q
                np.pi,  # R
                10000,  # pos_n
                10000,  # pos_e
                2300.0,  # alt
                100.0,  # pow
                5.0,  # nz_int
                5.0,  # ps_int
                5.0,  # nyr_int
            ], device=self.device
        )

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([3.0, 2.0, 1.0], device=self.device)
        lower_limit = torch.tensor([-1.0, -1.0, 0.0], device=self.device)

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        setpoint = self._setpoint
        bs = x.shape[0]

        # # speed limit
        # speed_limit = torch.logical_and(x[:, F16Tracking.VT] >= 400, x[:, F16Tracking.VT] <= 700)
        # goal_mask = torch.logical_and(goal_mask, speed_limit)
        #
        # # safe alpha and beta
        # alpha_safe = x[:, F16Tracking.ALPHA].abs() <= 0.5
        # goal_mask = torch.logical_and(goal_mask, alpha_safe)
        # beta_safe = x[:, F16Tracking.BETA].abs() <= 0.5
        # goal_mask = torch.logical_and(goal_mask, beta_safe)

        # head point to the goal
        pilot = WaypointAutopilot([tuple(setpoint.cpu().detach().numpy())], stdout=True)
        x_np = x.cpu().detach().numpy()
        psi_target = torch.zeros_like(x[:, F16Tracking.PSI])
        for i_batch in range(bs):
            psi_target[i_batch] = pilot.get_waypoint_data(x_np[i_batch, :])[0]
        heading_goal = torch.logical_and((x[:, F16Tracking.PSI] >= psi_target - 0.1).detach(),
                                         (x[:, F16Tracking.PSI] <= psi_target + 0.1).detach())
        goal_mask = torch.logical_and(goal_mask, heading_goal)

        # # small pitch and roll angle
        # pitch_low = x[:, F16Tracking.THETA].abs() <= 0.1
        # goal_mask = torch.logical_and(goal_mask, pitch_low)
        # roll_low = x[:, F16Tracking.PHI].abs() <= 0.2
        # goal_mask = torch.logical_and(goal_mask, roll_low)
        #
        # # low angular change rate
        # roll_rate_low = torch.logical_and(x[:, F16Tracking.Proll] >= -0.2,
        #                                   x[:, F16Tracking.Proll] <= torch.pi)
        # goal_mask = torch.logical_and(goal_mask, roll_rate_low)
        # pitch_rate_low = x[:, F16Tracking.Q].abs() <= 0.3
        # goal_mask = torch.logical_and(goal_mask, pitch_rate_low)
        # yaw_rate_low = x[:, F16Tracking.R].abs() <= 0.3
        # goal_mask = torch.logical_and(goal_mask, yaw_rate_low)
        #
        # # goal reaching
        # goal_pose = torch.logical_and(x[:, F16Tracking.POSE] >= setpoint[0] - 1000,
        #                               x[:, F16Tracking.POSE] <= setpoint[0] + 1000)
        # goal_mask = torch.logical_and(goal_mask, goal_pose)
        # goal_posn = torch.logical_and(x[:, F16Tracking.POSN] >= setpoint[1] - 1000,
        #                               x[:, F16Tracking.POSN] <= setpoint[1] + 1000)
        # goal_mask = torch.logical_and(goal_posn, goal_mask)
        goal_height = torch.logical_and(x[:, F16Tracking.H] >= setpoint[2] - 30,
                                        x[:, F16Tracking.H] <= setpoint[2] + 30)
        goal_mask = torch.logical_and(goal_mask, goal_height)

        # # power limit
        # goal_pow = torch.logical_and(x[:, F16Tracking.POW] >= 5,
        #                              x[:, F16Tracking.POW] <= 10)
        # goal_mask = torch.logical_and(goal_mask, goal_pow)

        return goal_mask

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # The F16 model is not batched, so we need to derivatives for each x separately
        n_batch = x.size()[0]
        xdot = torch.zeros_like(x).type_as(x)

        # Convert input to numpy
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()

        # modify u_np to fit the F16 model
        u_np_0 = u_np[:, :2]
        u_np_1 = np.zeros((n_batch, 1), dtype=u_np.dtype)
        u_np_2 = np.expand_dims(u_np[:, 2], axis=1)
        u_np = np.concatenate((u_np_0, u_np_1, u_np_2), axis=1)

        for batch in range(n_batch):
            # Compute derivatives at this point
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot_np, _, _, _, _ = controlled_f16(
                t, x_np[batch, :], u_np[batch, :], llc, f16_model=model
            )

            xdot[batch, :] = torch.tensor(xdot_np[:self.n_dims]).type_as(x)

        return xdot

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        # the nominal controller is supposed to track a point that is not exactly the setpoint
        setpoint = self._setpoint.cpu().numpy() + np.array([300, -200, 0])
        pilot = WaypointAutopilot([tuple(setpoint)], stdout=True)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        bs = x.shape[0]
        u = torch.zeros((bs, self.n_controls)).type_as(x)

        x_np = x.cpu().detach().numpy()
        for i_batch in range(bs):
            # get to the desired altitude
            nz_cmd = pilot.track_altitude(x_np[i_batch, :])

            # get the desired speed
            throttle = pilot.track_airspeed(x_np[i_batch, :])

            # point to the waypoint
            psi_cmd = pilot.get_waypoint_data(x_np[i_batch, :])[0]
            phi_cmd = pilot.get_phi_to_track_heading(x_np[i_batch, :], psi_cmd)
            ps_cmd = pilot.track_roll_angle(x_np[i_batch, :], phi_cmd)

            u[i_batch, :] = torch.tensor([nz_cmd, ps_cmd, throttle]).type_as(u)

        return u

    @property
    def use_lqr(self):
        return False

    @property
    def goal_point(self) -> torch.Tensor:
        return torch.tensor([
            540.0,  # vt
            deg2rad(2.1215),  # alpha
            0.0,  # beta
            0.0,  # phi
            0.0,  # theta
            torch.atan(self._setpoint[0] / self._setpoint[1]),  # psi
            0.0,  # P
            0.0,  # Q
            0.0,  # R
            self._setpoint[1],  # pos_n
            self._setpoint[0],  # pos_e
            self._setpoint[2],  # alt
            9.0,  # pow
            0.0,  # nz_int
            0.0,  # ps_int
            0.0,  # nyr_int
        ], dtype=torch.float, device=self.device).unsqueeze(0)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        setpoint = self._setpoint
        upper_limit, lower_limit = self.state_limits

        # specify goal conditions
        goal_conditions = torch.tensor([
            (lower_limit[F16Tracking.VT], upper_limit[F16Tracking.VT]),  # vt
            (lower_limit[F16Tracking.ALPHA], upper_limit[F16Tracking.ALPHA]),  # alpha
            (lower_limit[F16Tracking.BETA], upper_limit[F16Tracking.BETA]),  # beta
            (lower_limit[F16Tracking.PHI], upper_limit[F16Tracking.PHI]),  # phi
            (lower_limit[F16Tracking.THETA], upper_limit[F16Tracking.THETA]),  # theta
            (-0.1, 0.1),  # psi
            (lower_limit[F16Tracking.Proll], upper_limit[F16Tracking.Proll]),  # P
            (lower_limit[F16Tracking.Q], upper_limit[F16Tracking.Q]),  # Q
            (lower_limit[F16Tracking.R], upper_limit[F16Tracking.R]),  # R
            (lower_limit[F16Tracking.POSN], upper_limit[F16Tracking.POSN]),  # PN
            (lower_limit[F16Tracking.POSE], upper_limit[F16Tracking.POSE]),  # PE
            (setpoint[2], setpoint[2]),  # H
            (lower_limit[F16Tracking.POW], upper_limit[F16Tracking.POW]),  # pow
            (lower_limit[13], upper_limit[13]),  # integrator state 1
            (lower_limit[14], upper_limit[14]),  # integrator state 2
            (lower_limit[15], upper_limit[15]),  # integrator state 3
        ], dtype=torch.float, device=self.device)
        goal_points = torch.rand(batch_size, self.n_dims, device=self.device)
        goal_points = goal_points * (goal_conditions[:, 1] - goal_conditions[:, 0]) + goal_conditions[:, 0]

        # setup psi to make head point to the goal
        goal_points_np = goal_points.cpu().detach().numpy()
        pilot = WaypointAutopilot([tuple(setpoint.cpu().detach().numpy())], stdout=True)
        psi_target = torch.zeros_like(goal_points[:, F16Tracking.PSI])
        for i_batch in range(batch_size):
            psi_target[i_batch] = pilot.get_waypoint_data(goal_points_np[i_batch, :])[0]
        goal_points[:, F16Tracking.PSI] = psi_target

        return goal_points
