import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model.controller.neural_controller import NeuralController
from model.buffer import DemoBuffer


class BCTrainer:
    """
    Trainer for Behavior Cloning

    Parameters
    ----------
    policy: NeuralController
        neural network policy
    demo_buffer: DemoBuffer
        buffer of demonstrations
    lr: float
        learning rate
    """

    def __init__(self, policy: NeuralController, demo_buffer: DemoBuffer, lr: float = 3e-4):
        self.policy = policy
        self.demo_buffer = demo_buffer
        self.lr = lr
        self.optim = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-3)
        self.loss_fun = nn.MSELoss()

    def train(self, n_iter: int, batch_size: int):
        for i_iter in tqdm(range(n_iter), ncols=80):
            states, exp_actions = self.demo_buffer.sample_policy(batch_size)
            actions = self.policy(states)
            loss = self.loss_fun(actions, exp_actions)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if i_iter % int(n_iter / 10) == 0 or i_iter == n_iter - 1:
                tqdm.write(f'iter: {i_iter}, loss: {loss.item():.2e}')
