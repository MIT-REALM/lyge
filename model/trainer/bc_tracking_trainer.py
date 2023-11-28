import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model.controller.neural_tracking_controller import NeuralTrackingController
from model.buffer import TrackingBuffer


class BCTrackingTrainer:
    """
    Trainer for Behavior Cloning for tracking controllers

    Parameters
    ----------
    policy: NeuralController
        neural network policy
    demo_buffer: TrackingBuffer
        buffer of demonstrations
    lr: float
        learning rate
    """

    def __init__(self, policy: NeuralTrackingController, demo_buffer: TrackingBuffer, lr: float = 3e-4):
        self.policy = policy
        self.demo_buffer = demo_buffer
        self.lr = lr
        self.optim = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-3)
        self.loss_fun = nn.MSELoss()

    def train(self, n_iter: int, batch_size: int):
        for i_iter in tqdm(range(n_iter), ncols=80):
            states, exp_actions, states_ref, actions_ref = self.demo_buffer.sample_policy(batch_size)
            actions = self.policy(states, states_ref, actions_ref.unsqueeze(-1)).squeeze(-1)
            loss = self.loss_fun(actions, exp_actions)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if i_iter % int(n_iter / 10) == 0 or i_iter == n_iter - 1:
                tqdm.write(f'iter: {i_iter}, loss: {loss.item():.2e}')
