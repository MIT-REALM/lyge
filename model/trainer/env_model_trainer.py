import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..env_model.base import NeuralEnv
from ..buffer import DemoBuffer
from ..env.base import ControlAffineSystem


class EnvModelTrainer:
    """
    trainer of the env model

    Parameters
    ----------
    env_model: EnvModel
        initialized env model
    demo_buffer: DemoBuffer
        preloaded buffer of demonstrations
    lr: float
        learning rate
    """

    def __init__(
            self,
            env_model: NeuralEnv,
            env: ControlAffineSystem,
            demo_buffer: DemoBuffer,
            writer: SummaryWriter,
            lr: float = 3e-4
    ):
        self.env_model = env_model
        self.env = env
        self.demo_buffer = demo_buffer
        self.writer = writer
        self.optim = optim.Adam(env_model.parameters(), lr=lr, weight_decay=1e-3)
        self.loss_fun = nn.MSELoss()

    def train(self, n_iter: int, batch_size: int, sample_demo=True, outer_iter: int = None):
        """
        train the env model

        Parameters
        ----------
        n_iter: int
            number of iterations
        batch_size: int
            batch size of sampling demonstrations
        sample_demo: bool
            if true, sample from the demo_buffer, else, sample from the state space
        outer_iter: int
            for summary writer
        """
        for i_iter in tqdm(range(n_iter), ncols=80):
            if sample_demo:
                states, actions, next_states = self.demo_buffer.sample_transition(batch_size)
            else:
                states = self.env.sample_states(batch_size)
                actions = self.env.sample_ctrls(batch_size)
                next_states = self.env.forward(states, actions)
            real_xdot = (next_states - states) / self.env_model.dt
            pred_xdot = self.env_model.closed_loop_dynamics(states, actions)

            loss = self.loss_fun(real_xdot, pred_xdot)

            if hasattr(self.env_model, 'sparse_g'):
                if self.env_model.sparse_g:
                    pred_g = self.env_model.control_affine_dynamics(states)[1]
                    loss += 1e-4 * torch.linalg.norm(pred_g.reshape(states.shape[0], -1), dim=1, ord=1).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if outer_iter is not None:
                self.writer.add_scalar('loss/env', loss.item(), i_iter + outer_iter * n_iter)
            else:
                self.writer.add_scalar('loss/env', loss.item(), i_iter)

            if i_iter % int(n_iter / 10) == 0 or i_iter == n_iter - 1:
                tqdm.write(f'iter: {i_iter}, loss: {loss:.2e}')
