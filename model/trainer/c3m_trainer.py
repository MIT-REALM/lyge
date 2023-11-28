import torch
import torch.optim as optim

from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter

from .utils import jacobian, weighted_grad

from model.controller.neural_c3m_controller import NeuralC3MController
from model.controller.neural_tracking_controller import NeuralTrackingController
from model.env_model.safe_tracking import NeuralSafeCriticalTrackingEnv
from model.env.base import ControlAffineSystem


class C3MTrainer:

    def __init__(
            self,
            controller: NeuralC3MController,
            writer: SummaryWriter,
            env: NeuralSafeCriticalTrackingEnv = None,
            gt_env: ControlAffineSystem = None,
            hyper_params: dict = None,
    ):
        self.controller = controller
        self.env = env
        self.writer = writer
        self.gt_env = gt_env
        self.hyper_params = hyper_params
        self.base_lr_ctrl = 3e-4
        self.base_lr_CCM = 3e-4
        self.optim_ctrl = optim.Adam(self.controller.controller.parameters(), lr=self.base_lr_ctrl, weight_decay=1e-3)
        self.optim_CCM = optim.Adam(self.controller.parameters(), lr=self.base_lr_CCM, weight_decay=1e-3)

        # set default hyper-parameters
        if hyper_params is None:
            self.hyper_params = {}

    def train(
            self,
            n_iter: int,
            batch_size: int,
            ref_controller: NeuralTrackingController = None,
            outer_iter: int = None,
            update_M: bool = False,
            use_gt_env: bool = False,
            eval_iter: int = None,
    ):
        if use_gt_env:
            assert self.gt_env is not None
        else:
            assert self.env is not None

        if eval_iter is None:
            eval_iter = n_iter / 10

        for i_iter in tqdm(range(n_iter), ncols=80):
            # sample states
            if use_gt_env:
                states, states_ref = self.gt_env.sample_states_ref(batch_size)
                ctrls_ref = self.gt_env.sample_ctrls(batch_size)
            else:
                demo_batch_size = int(batch_size * 0.7)
                sample_batch_size = batch_size - demo_batch_size
                safe_states, safe_states_ref = self.env.sample_safe_states_ref(demo_batch_size)
                safe_ctrls_ref = self.env.sample_safe_ctrls(demo_batch_size)
                sample_states, sample_states_ref = self.env.sample_states_ref(sample_batch_size)
                sample_ctrls_ref = self.env.sample_ctrls(sample_batch_size)
                states = torch.cat([safe_states, sample_states])
                states_ref = torch.cat([safe_states_ref, sample_states_ref])
                ctrls_ref = torch.cat([safe_ctrls_ref, sample_ctrls_ref])

            states.unsqueeze_(-1)
            states_ref.unsqueeze_(-1)
            ctrls_ref.unsqueeze_(-1)
            ctrls_prev = None
            if ref_controller is not None:
                with torch.no_grad():
                    ctrls_prev = ref_controller(states, states_ref, ctrls_ref)

            states.requires_grad = True
            states_ref.requires_grad = True
            ctrls_ref.requires_grad = True

            evaluate = True if i_iter % eval_iter == 0 else False
            loss, evaluation = self.calculate_loss(
                states,
                states_ref,
                ctrls_ref,
                use_gt_env,
                u_prev=ctrls_prev,
                evaluate=evaluate,
                update_M=update_M
            )

            total_loss = torch.tensor(0.).type_as(states)
            for value in loss.values():
                total_loss += value
            self.optim_ctrl.zero_grad()
            self.optim_CCM.zero_grad()
            total_loss.backward()
            self.optim_ctrl.step()
            self.optim_CCM.step()

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
                    self.writer.add_scalar(f'evaluation/{eval_name}',
                                           evaluation[eval_name].item(), i_iter + outer_iter * n_iter)
                else:
                    self.writer.add_scalar(f'evaluation/{eval_name}', evaluation[eval_name].item(), i_iter)

            if i_iter % int(n_iter / 10) == 0 or i_iter == n_iter - 1:
                tqdm.write(f'iter: {i_iter}, loss: {total_loss.item():.2e}')

            if evaluate:
                verbose = f'iter: {i_iter}'
                for eval_name in evaluation.keys():
                    verbose += f', {eval_name}: {evaluation[eval_name].item():.2f}'
                tqdm.write(verbose)

    def calculate_loss(
            self,
            x: torch.Tensor,
            x_ref: torch.Tensor,
            u_ref: torch.Tensor,
            use_gt_env: bool,
            u_prev: torch.Tensor = None,
            evaluate: bool = False,
            update_M: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        bs = x.shape[0]
        loss = {}
        evaluation = {}

        W = self.controller.W(x)
        M = torch.inverse(W)
        if use_gt_env:
            f, g = self.gt_env.control_affine_dynamics(x)
        else:
            f, g = self.env.control_affine_dynamics(x.squeeze(-1))

        u = self.controller.u(x, x_ref, u_ref)
        xdot = f + torch.bmm(g, u)
        K = jacobian(u, x)
        g_bot = self.controller.g_bot(x)

        DfDx = jacobian(f, x)
        DgDx = torch.zeros(
            bs, self.controller.state_dim, self.controller.state_dim, self.controller.action_dim).type_as(x)
        for i in range(self.controller.action_dim):
            DgDx[:, :, :, i] = jacobian(g[:, :, i].unsqueeze(-1), x)

        A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DgDx[:, :, :, i]
                        for i in range(self.controller.action_dim)])
        Mdot = weighted_grad(M, xdot, x, detach=not update_M)

        if update_M:
            CCM = Mdot + torch.bmm((A + torch.bmm(g, K)).transpose(1, 2), M)\
              + torch.bmm(M, A + torch.bmm(g, K)) + 2 * self.controller.c3m_lambda * M
        else:
            CCM = Mdot + torch.bmm((A + torch.bmm(g, K)).transpose(1, 2), M.detach()) \
                  + torch.bmm(M.detach(), A + torch.bmm(g, K)) + 2 * self.controller.c3m_lambda * M.detach()

        # C1 loss
        C1_inner = -weighted_grad(W, f, x) + torch.bmm(DfDx, W) + torch.bmm(W, DfDx.transpose(1, 2))\
                   + 2 * self.controller.c3m_lambda * W
        C1 = torch.bmm(torch.bmm(g_bot.transpose(1, 2), C1_inner), g_bot)

        # C2 loss
        C2 = []
        for i in range(self.controller.action_dim):
            C2_inner = weighted_grad(W, g[:, :, i].unsqueeze(-1), x)\
                       - (torch.bmm(DgDx[:, :, :, i], W) + torch.bmm(W, DgDx[:, :, :, i].transpose(1, 2)))
            C2.append(torch.bmm(torch.bmm(g_bot.transpose(1, 2), C2_inner), g_bot))

        # total loss
        eps = 0.1 * self.controller.c3m_lambda
        CCM_eig = torch.linalg.eigh(CCM + eps * torch.eye(CCM.shape[-1]).type_as(CCM))[0]
        loss['CCM ND'] = torch.relu(CCM_eig).mean()
        C1_eig = torch.linalg.eigh(C1 + eps * torch.eye(C1.shape[-1]).type_as(CCM))[0]
        loss['C1 ND'] = torch.relu(C1_eig).mean()
        W_eig = torch.linalg.eigh(-self.controller.w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type_as(W) + W)[0]
        loss['W ub'] = torch.relu(W_eig).mean()
        loss['C2'] = sum((c2**2).reshape(bs, -1).sum(dim=1).mean() for c2 in C2)
        if u_prev is not None:
            loss['controller deviation'] = 1e-3 * ((u - u_prev) ** 2).mean()

        if evaluate:
            evaluation['CCM ND'] = ((torch.linalg.eigh(CCM)[0] >= 0).sum(dim=1) == 0).sum() / bs
            evaluation['C1 ND'] = ((torch.linalg.eigh(C1)[0] >= 0).sum(dim=1) == 0).sum() / bs

        return loss, evaluation
