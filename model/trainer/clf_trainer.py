import torch
import torch.optim as optim

from tqdm import tqdm
from typing import Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from model.buffer import DemoBuffer
from model.controller import Controller, NeuralCLFController
from model.env.base import ControlAffineSystem
from model.env_model.safe_affine import NeuralSafeCriticalAffineEnv
from model.env_model.safe_critical import NeuralSafeCriticalEnv
from model.env_model.base import NeuralEnv

import pdb


class CLFTrainer:

    def __init__(
            self,
            controller: NeuralCLFController,
            controller_ref: Controller,
            env: NeuralEnv,
            writer: SummaryWriter,
            demo_buffer: DemoBuffer,
            lr_lyapunov: float = 3e-4,
            lr_controller: float = 5e-4,
            baseline: Controller = None,
            gt_env: ControlAffineSystem = None,
            hyper_params: dict = None
    ):
        self.controller = controller
        self.controller_ref = controller_ref
        self.baseline_ref = baseline
        self.env = env
        self.demo_buffer = demo_buffer
        self.gt_env = gt_env
        self.writer = writer
        self.lr_lyapunov = lr_lyapunov
        self.lr_controller = lr_controller
        self.optim_lyapunov = optim.Adam(self.controller.lyapunov.parameters(), lr=lr_lyapunov, weight_decay=1e-3)
        self.optim_controller = optim.Adam(self.controller.controller.parameters(), lr=lr_controller, weight_decay=1e-3)
        self.hyper_params = hyper_params

        # set default hyper-parameters
        if hyper_params is None:
            self.hyper_params = {
                'loss_coefs': {
                    'goal': 2e1,
                    'positive': 1e5,
                    'decrease': 5e1,
                    'control': 2e-1,
                },
                'loss_eps': {
                    'positive': 0.1,
                    'decrease': 2.0,
                }
            }

    def train(
            self,
            n_iter: int,
            batch_size: int,
            sample_demo: bool = True,
            global_sample: bool = False,
            outer_iter: int = None,
    ):
        """
        Train the CLF controller

        Parameters
        ----------
        n_iter: int
            number of iterations
        batch_size: int
            batch size
        sample_demo: bool
            if true, sample from the demo_buffer, else, sample from the state space
        global_sample: bool
            if true, also sample from the state space
        outer_iter: Optional[int]
            for summary writer
        """
        # jointly train the lyapunov function and the controller
        for i_iter in tqdm(range(1, n_iter), ncols=80):
            # sample normalized states
            if isinstance(self.env, NeuralSafeCriticalAffineEnv) or isinstance(self.env, NeuralSafeCriticalEnv):
                if sample_demo:
                    if global_sample:
                        demo_batch_size = int(batch_size * 0.8)
                        sample_batch_size = batch_size - demo_batch_size
                        safe_states = self.env.sample_safe_states(demo_batch_size)
                        sample_states = self.env.sample_states(sample_batch_size)
                        states = torch.cat([safe_states, sample_states])
                    else:
                        states = self.env.sample_safe_states(batch_size)
                else:
                    states = self.env.sample_states(batch_size)
            elif isinstance(self.env, ControlAffineSystem):
                states = self.env.sample_states(batch_size)
            else:
                raise KeyError('Wrong env type')

            loss = {}
            loss.update(self.boundary_loss(
                states,
                self.env.goal_mask(states),
            ))
            decent_loss, evaluation = self.descent_loss(states, self.controller_ref(states))
            loss.update(decent_loss)

            total_loss = torch.tensor(0.0).type_as(states)
            for value in loss.values():
                total_loss += value
            if torch.isnan(total_loss):
                pdb.set_trace()

            self.optim_lyapunov.zero_grad()
            self.optim_controller.zero_grad()
            total_loss.backward()
            self.optim_lyapunov.step()
            self.optim_controller.step()

            for loss_name in loss.keys():
                if outer_iter is not None:
                    self.writer.add_scalar(f'loss/{loss_name}', loss[loss_name].item(), i_iter + outer_iter * n_iter)
                else:
                    self.writer.add_scalar(f'loss/{loss_name}', loss[loss_name].item(), i_iter)
            if outer_iter is not None:
                self.writer.add_scalar(f'loss/total loss', total_loss.item(), i_iter + outer_iter * n_iter)
            else:
                self.writer.add_scalar(f'loss/total loss', total_loss.item(), i_iter)

            for eval_name in evaluation.keys():
                if outer_iter is not None:
                    self.writer.add_scalar(f'eval/{eval_name}', evaluation[eval_name].item(),
                                           i_iter + outer_iter * n_iter)
                else:
                    self.writer.add_scalar(f'eval/{eval_name}', evaluation[eval_name].item(), i_iter)

            if i_iter % int(n_iter / 10) == 0 or i_iter == n_iter - 1:
                tqdm.write(f'iter: {i_iter}, loss: {total_loss.item():.2e}')

    def boundary_loss(
            self,
            x: torch.Tensor,
            goal_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the CLF boundary loss according to the following requirements:
            1.) CLF should be minimized on the goal point
            2.) V >= 0 in all region

        Parameters
        ----------
        x: torch.Tensor
            input states (normalized)
        goal_mask: torch.Tensor
            same dimension as x, 1 is in the goal region and 0 is not

        Returns
        -------
        loss: Dict[str, torch.Tensor]
            dict of loss terms, including
                1.) CLF goal term
                2.) CLF positive term
        """
        loss = {}
        V = self.controller.V(x)

        # CLF should be minimized on the goal point
        V_goal_1 = self.controller.V(self.env.goal_point)
        V_goal_2 = V[goal_mask]
        # V_goal_2 = self.controller.V(self.env.sample_goal(batch_size=64))
        V_goal = torch.cat([V_goal_1, V_goal_2], dim=0)
        goal_term = (V_goal ** 2).mean()
        loss['CLBF goal term'] = self.hyper_params['loss_coefs']['goal'] * goal_term
        if torch.isnan(loss['CLBF goal term']):
            aaa = 0
            pdb.set_trace()

        return loss

    def descent_loss(
            self,
            x: torch.Tensor,
            u_ref: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute the CLF decent loss. The CLF decrease condition requires that V is decreasing
        everywhere. We'll encourage this in two ways:
            1.) Compute the CLF decrease at each point by linearizing
            2.) Compute the CLF decrease at each point by simulating

        Parameters
        ----------
        x: torch.Tensor
            input states
        u_ref: torch.Tensor
            control signal of the reference nominal controller

        Returns
        -------
        loss: Dict[str, torch.Tensor]
            dict of loss terms, including:
                1.) QP relaxation
                2.) CLF descent term (linearized)
                3.) CLF descent term (simulated)
        """
        loss = {}
        evaluation = {}
        x.requires_grad = True
        V, JV = self.controller.V_with_Jacobian(x)
        u = self.controller.u(x)
        xdot = self.env.closed_loop_dynamics(x, u)

        # Now compute the decrease using linearization
        eps = self.hyper_params['loss_eps']['decrease']
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        # Get the current value of the CLF and its Lie derivatives
        Vdot = torch.bmm(JV, xdot.unsqueeze(-1)).squeeze(1)
        violation = torch.relu(eps + Vdot + self.controller.clf_lambda * V)
        clbf_descent_term_lin += violation.mean()
        loss['CLBF descent term (linearized)'] = self.hyper_params['loss_coefs']['decrease'] * clbf_descent_term_lin
        if torch.isnan(loss['CLBF descent term (linearized)']):
            aaa = 0
            pdb.set_trace()

        # Now compute the decrease using simulation
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        x_next = x + xdot * self.env.dt
        V_next = self.controller.V(x_next)
        violation = torch.relu(eps + ((V_next - V) / self.env.dt) + self.controller.clf_lambda * V)
        clbf_descent_term_sim += violation.mean()
        loss['CLBF descent term (simulated)'] = self.hyper_params['loss_coefs']['decrease'] * clbf_descent_term_sim
        if torch.isnan(loss['CLBF descent term (simulated)']):
            aaa = 0
            pdb.set_trace()
        with torch.no_grad():
            real_violation = torch.relu((V_next - V) / self.env.dt + self.controller.clf_lambda * V)
        evaluation['CLF descent violation ratio'] = torch.count_nonzero(real_violation) / real_violation.shape[0]

        # evaluate the ground truth decent violation ratio
        if self.gt_env is not None:
            with torch.no_grad():
                x_next_gt = self.gt_env.forward(x, u)
                V_next_gt = self.controller.V(x_next_gt)
                violation_gt = torch.relu((V_next_gt - V) / self.env.dt + self.controller.clf_lambda * V)
                evaluation['GT CLF decent violation ratio'] = torch.count_nonzero(violation_gt) / violation_gt.shape[0]

        # compute the loss of deviation from the reference controller
        loss['Controller deviation'] = self.hyper_params['loss_coefs']['control'] * ((u - u_ref) ** 2).mean()
        if torch.isnan(loss['Controller deviation']):
            aaa = 0
            pdb.set_trace()
        if self.baseline_ref is not None:
            u_baseline = self.baseline_ref(x)
            loss['Baseline deviation'] = self.hyper_params['loss_coefs']['control'] * ((u - u_baseline) ** 2).mean()
            if torch.isnan(loss['Baseline deviation']):
                aaa = 0
                pdb.set_trace()

        return loss, evaluation
