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

import model.env.aerobench as aerobench_loader  # type: ignore
from aerobench.highlevel.controlled_f16 import controlled_f16  # type: ignore
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot  # type: ignore
from aerobench.lowlevel.low_level_controller import LowLevelController  # type: ignore
from aerobench.visualize.anim3d import get_script_path  # type: ignore
from aerobench.visualize import plot  # type: ignore


class F16GCAS(ControlAffineSystem):
    """
    F16 ground collision avoidance system.

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
        u[2] = side acceleration + yaw rate (usually regulated to 0)
        u[3] = throttle command (0.0, 1.0)

    The system is parameterized by
        lag_error: the additive error in the engine lag state dynamics
    """

    # number of states and controls
    N_DIMS = 16
    N_CONTROLS = 4

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
    U_NYR = 2  # desired side acceleration + yaw rate
    U_THROTTLE = 3  # throttle command

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
            dt: float = 0.02,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None,
    ):
        super(F16GCAS, self).__init__(device, dt, params, controller_dt)

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        self.P = torch.eye(self.n_dims)

    def reset(self) -> torch.Tensor:
        initial_conditions = torch.tensor([
            (520.0, 560.0),  # vt
            (deg2rad(2.1215), deg2rad(2.1215)),  # alpha
            (-0.0, 0.0),  # beta
            (0.0, 0.0),  # phi
            ((-math.pi / 2) * 0.7, (-math.pi / 2) * 0.7),  # theta
            (0.8 * math.pi, 0.8 * math.pi),  # psi
            (-5.0, 5.0),  # P
            (-1.0, 1.0),  # Q
            (-1.0, 1.0),  # R
            (-0.0, 0.0),  # PN
            (-0.0, 0.0),  # PE
            (2600.0, 3000.0),  # H
            (4.0, 5.0),  # pow
            (0.0, 0.0),  # integrator state 1
            (0.0, 0.0),  # integrator state 2
            (0.0, 0.0),  # integrator state 3
        ], dtype=torch.float, device=self.device)
        self._t = 0
        self._state = torch.rand(1, self.n_dims, device=self.device)
        self._state = self._state * (initial_conditions[:, 1] - initial_conditions[:, 0]) + initial_conditions[:, 0]
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        if u.ndim == 1:
            u = u.unsqueeze(0)

        # modify angles
        # while self._state[0, 5] > np.pi:
        #     self._state[0, 5] -= 2 * np.pi
        # while self._state[0, 5] < -np.pi:
        #     self._state[0, 5] += 2 * np.pi
        # while self._state[0, 3] > np.pi:
        #     self._state[0, 3] -= 2 * np.pi
        # while self._state[0, 3] < -np.pi:
        #     self._state[0, 3] += 2 * np.pi

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
        # if done:
        #     if (self.state < lower_x_lim).any():
        #         violation = torch.nonzero((self.state < lower_x_lim).detach()).squeeze(-1)
        #         for v in violation:
        #             tqdm.write(f'F16 error: violate lower bound: {self.state_name(v)}, '
        #                        f'value: {self.state[v]:.2f}, time: {self._t}')
        #     if (self.state > upper_x_lim).any():
        #         violation = torch.nonzero((self.state > upper_x_lim).detach()).squeeze(-1)
        #         for v in violation:
        #             tqdm.write(f'F16 error: violate upper bound: {self.state_name(v)}, '
        #                        f'value: {self.state[v]:.2f}, time: {self._t}')
        if self.state[F16GCAS.H] > 1200:
            reward = float(2. - (self.state[F16GCAS.H] - 1200.) * 0.001)
        elif self.state[F16GCAS.H] < 800:
            reward = float(2. - (800. - self.state[F16GCAS.H]) * 0.001 * 2)
        else:
            reward = 2.

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
        dx = state[F16GCAS.POSE]
        dy = state[F16GCAS.POSN]
        dz = state[F16GCAS.H]

        ax.set_xlabel('X [ft]', fontsize=14)
        ax.set_ylabel('Y [ft]', fontsize=14)
        ax.set_zlabel('Altitude [ft]', fontsize=14)

        # text
        fontsize = 14
        time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
        alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
        v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
        beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

        time_text.set_text(f't = {self._t}')
        alt_text.set_text(f'h = {state[F16GCAS.H]:.2f} ft')
        v_text.set_text(f'V = {state[F16GCAS.VT]:.2f} ft/sec')
        alpha_text.set_text(f'$\\alpha$ = {rad2deg(state[F16GCAS.ALPHA]):.2f} deg')
        beta_text.set_text(f'$\\beta$ = {rad2deg(state[F16GCAS.BETA]):.2f} deg')
        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(
            rad2deg(state[F16GCAS.PHI]), rad2deg(state[F16GCAS.THETA]), rad2deg(state[F16GCAS.PSI])))

        # set the space for F16
        s = 30
        pts = scale3d(f16_pts, [-s, s, s])
        pts = rotate3d(pts, state[F16GCAS.THETA], state[F16GCAS.PSI] - math.pi / 2, -state[F16GCAS.PHI])
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
        dx = state[F16GCAS.POSE]
        dy = state[F16GCAS.POSN]
        dz = state[F16GCAS.H]

        ax.set_xlabel('X [ft]', fontsize=14)
        ax.set_ylabel('Y [ft]', fontsize=14)
        ax.set_zlabel('Altitude [ft]', fontsize=14)

        # text
        fontsize = 14
        time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
        alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
        v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
        beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')
        ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

        time_text.set_text(f't = {t}')
        alt_text.set_text(f'h = {state[F16GCAS.H]:.2f} ft')
        v_text.set_text(f'V = {state[F16GCAS.VT]:.2f} ft/sec')
        alpha_text.set_text(f'$\\alpha$ = {rad2deg(state[F16GCAS.ALPHA]):.2f} deg')
        beta_text.set_text(f'$\\beta$ = {rad2deg(state[F16GCAS.BETA]):.2f} deg')
        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(
            rad2deg(state[F16GCAS.PHI]), rad2deg(state[F16GCAS.THETA]), rad2deg(state[F16GCAS.PSI])))

        # set the space for F16
        s = 30
        pts = scale3d(f16_pts, [-s, s, s])
        pts = rotate3d(pts, state[F16GCAS.THETA], state[F16GCAS.PSI] - math.pi / 2, -state[F16GCAS.PHI])
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

        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        return {"lag_error": 0.0}

    def validate_params(self, params: dict) -> bool:
        valid = "lag_error" in params

        return valid

    def state_name(self, dim: int) -> str:
        return F16GCAS.STATE_NAME[dim]

    def distance2goal(self, state: torch.Tensor = None):
        if state is None:
            state = self.state
        assert state.ndim == 1
        return float(state[F16GCAS.H])
        # return abs(float(state[F16GCAS.H] - 1200))
        # h = state[F16GCAS.H]
        # if h > 1600:
        #     return h - 1600.
        # elif h < 600:
        #     return 600. - h
        # else:
        #     return 0.

    @property
    def n_dims(self) -> int:
        return F16GCAS.N_DIMS

    @property
    def n_controls(self) -> int:
        return F16GCAS.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return F16GCAS.MAX_EPISODE_STEPS

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
                -2 * np.pi,  # P
                -2 * np.pi,  # Q
                -2 * np.pi,  # R
                -10000,  # pos_n
                -10000,  # pos_e
                0.0,  # alt
                0.0,  # pow
                -20.0,  # nz_int
                -20.0,  # ps_int
                -20.0,  # nyr_int
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
                2 * np.pi,  # P
                2 * np.pi,  # Q
                2 * np.pi,  # R
                10000,  # pos_n
                10000,  # pos_e
                3500.0,  # alt
                30.0,  # pow
                20.0,  # nz_int
                20.0,  # ps_int
                20.0,  # nyr_int
            ], device=self.device
        )

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([6.0, 20.0, 20.0, 1.0], device=self.device)
        lower_limit = torch.tensor([-1.0, -20.0, -20.0, 0.0], device=self.device)

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # nice posture
        nose_high_enough = x[:, F16GCAS.THETA] + x[:, F16GCAS.ALPHA] >= 0.0
        goal_mask = torch.logical_and(goal_mask, nose_high_enough)
        wings_near_level = x[:, F16GCAS.PHI].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, wings_near_level)

        # low angular change rate
        roll_rate_low = x[:, F16GCAS.Proll].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, roll_rate_low)
        pitch_rate_low = x[:, F16GCAS.Q].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, pitch_rate_low)
        yaw_rate_low = x[:, F16GCAS.R].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, yaw_rate_low)

        # angle should be near 0
        roll_angle_small = x[:, F16GCAS.PHI].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, roll_angle_small)
        pitch_angle_small = x[:, F16GCAS.THETA].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, pitch_angle_small)
        yaw_angle_small = x[:, F16GCAS.PSI].abs() <= 0.1
        goal_mask = torch.logical_and(goal_mask, yaw_angle_small)

        # safe alpha and beta
        alpha_safe = x[:, F16GCAS.ALPHA].abs() <= 0.5
        goal_mask = torch.logical_and(goal_mask, alpha_safe)
        beta_safe = x[:, F16GCAS.BETA].abs() <= 0.5
        goal_mask = torch.logical_and(goal_mask, beta_safe)

        # speed limit
        speed_limit = torch.logical_and(x[:, F16GCAS.VT] >= 500, x[:, F16GCAS.VT] <= 600)
        goal_mask = torch.logical_and(goal_mask, speed_limit)

        # power limit
        power_limit = torch.logical_and(x[:, F16GCAS.POW] >= 3, x[:, F16GCAS.POW] <= 6)
        goal_mask = torch.logical_and(goal_mask, power_limit)

        # good height
        above_deck = x[:, F16GCAS.H] >= 1000.0
        goal_mask = torch.logical_and(goal_mask, above_deck)
        below_threshold = x[:, F16GCAS.H] <= 1200.0
        goal_mask = torch.logical_and(goal_mask, below_threshold)

        return goal_mask

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.params

        # The f16 model is not batched, so we need to compute f and g for each row of x
        n_batch = x.shape[0]
        f = torch.zeros((n_batch, self.n_dims, 1)).type_as(x)
        g = torch.zeros(n_batch, self.n_dims, self.n_controls).type_as(x)

        # Convert input to numpy
        x = x.detach().cpu().numpy()
        for batch in range(n_batch):
            # Get the derivatives at each of n_controls + 1 linearly independent points
            # (plus zero) to fit control-affine dynamics
            u = np.zeros((1, self.n_controls))
            for i in range(self.n_controls):
                u_i = np.zeros((1, self.n_controls))
                u_i[0, i] = 1.0
                u = np.vstack((u, u_i))

            # Compute derivatives at each of these points
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot = np.zeros((self.n_controls + 1, self.n_dims))
            for i in range(self.n_controls + 1):
                xdot[i, :], _, _, _, _ = controlled_f16(
                    t, x[batch, :], u[i, :], llc, f16_model=model
                )

            # Run a least-squares regression to fit control-affine dynamics
            # We want a relationship of the form
            #       xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
            # Augment the inputs with a one column for the control-independent part
            regressors = np.hstack((np.ones((self.n_controls + 1, 1)), u))
            # Compute the least-squares fit and find A^T such that xdot = [1, u] A^T
            A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
            A = A.T
            # Extract the control-affine fit
            f[batch, :, 0] = torch.tensor(A[:, 0]).type_as(f)
            g[batch, :, :] = torch.tensor(A[:, 1:]).type_as(g)

            # Add in the lag error (which we're treating as bounded additive error)
            f[batch, self.POW] += params["lag_error"]

        return f, g

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # The F16 model is not batched, so we need to derivatives for each x separately
        n_batch = x.size()[0]
        xdot = torch.zeros_like(x).type_as(x)

        # Convert input to numpy
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        for batch in range(n_batch):
            # Compute derivatives at this point
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot_np, _, _, _, _ = controlled_f16(
                t, x_np[batch, :], u_np[batch, :], llc, f16_model=model
            )

            xdot[batch, :] = torch.tensor(xdot_np).type_as(x)

        return xdot

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        gcas = GcasAutopilot()
        gcas.cfg_flight_deck = 1700

        # The autopilot is not meant to be run on batches, so we need to get control
        # inputs separately
        y = x
        if y.ndim == 1:
            y = y.unsqueeze(0)
        n_batch = y.size()[0]
        u = torch.zeros((n_batch, self.n_controls)).type_as(y)

        x_np = y.cpu().detach().numpy()
        for batch in range(n_batch):
            # The GCAS autopilot is implemented as a state machine that first rolls and
            # then pulls up. Here we unwrap the state machine logic to get a simpler
            # mapping from state to control

            # If the plane is not hurtling towards the ground, don't do anything
            if gcas.is_nose_high_enough(x_np[batch, :]) or gcas.is_above_flight_deck(
                    x_np[batch, :]
            ):
                continue

            # If we are hurtling towards the ground and the plane isn't level, we need
            # to roll to get level
            if not gcas.is_roll_rate_low(x_np[batch, :]) or not gcas.are_wings_level(
                    x_np[batch, :]
            ):
                u[batch, :] = torch.tensor(gcas.roll_wings_level(x_np[batch, :]))
                continue

            # If we are hurtling towards the ground and the plane IS level, then we need
            # to pull up
            u[batch, :] = torch.tensor(gcas.pull_nose_level()).type_as(u)

        return u

    @property
    def use_lqr(self):
        return False

    @property
    def goal_point(self) -> torch.Tensor:
        return torch.tensor([
            540.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1100.0,
            5.0,
            0.0,
            0.0,
            0.0,
        ], device=self.device).unsqueeze(0)

    @property
    def state(self) -> torch.Tensor:
        if self._state is not None:
            while self._state[0, 5] > np.pi:
                self._state[0, 5] -= np.pi * 2
            while self._state[0, 5] < -np.pi:
                self._state[0, 5] += np.pi * 2
            while self._state[0, 3] > np.pi:
                self._state[0, 3] -= np.pi * 2
            while self._state[0, 3] < -np.pi:
                self._state[0, 3] += np.pi * 2
            return self._state.squeeze(0)
        else:
            raise ValueError('State is not initialized')
