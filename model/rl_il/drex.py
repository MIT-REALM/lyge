import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from tqdm import tqdm
from torch.nn.utils.convert_parameters import parameters_to_vector

from .ppo import PPO
from .utils import NoisePreferenceDataset
from ..buffer import DemoBuffer
from ..env.base import ControlAffineSystem
from ..controller import NeuralController
from ..network.mlp import NormalizedMLP


class DREX(PPO):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            env: ControlAffineSystem = None,
            demo_buffer: DemoBuffer = None,
            policy_bc: NeuralController = None,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            batch_size_reward: int = 64,
            size_reward_dataset: int = 5000,
            n_reward_model: int = 3,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_reward: float = 1e-4,
            units_actor: tuple = (128, 128),
            units_critic: tuple = (128, 128),
            units_reward: tuple = (128, 128),
            epoch_ppo: int = 20,
            iter_reward: int = 10000,
            l2_ratio_reward: float = 0.01,
            noise_range: np.array = np.arange(0., 1.0, 0.05),
            n_demo_traj: int = 5,
            steps_exp: int = 40,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0
    ):
        super().__init__(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                         gamma, rollout_length, mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
                         epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # synthetic noisy dataset
        self.noisy_dataset = NoisePreferenceDataset(
            env=env,
            device=device,
            max_steps=50,
            min_margin=0.3,
        )
        self.env = env

        # expert's buffer
        self.demo_buffer = demo_buffer

        # BC policy
        self.bc = policy_bc

        if policy_bc is not None and demo_buffer is not None and env is not None:
            # reward function
            self.reward_funcs = []
            self.reward_optims = []
            for i in range(n_reward_model):
                self.reward_funcs.append(NormalizedMLP(
                    in_dim=state_dim,
                    out_dim=1,
                    input_mean=goal_point.squeeze(),
                    input_std=state_std,
                    hidden_layers=units_reward,
                    hidden_activation=nn.Tanh(),
                ).to(device))
                self.reward_optims.append(Adam(self.reward_funcs[i].parameters(), lr=lr_reward))
            self.batch_size_reward = batch_size_reward
            self.iter_reward = iter_reward
            self.l2_ratio_reward = l2_ratio_reward
            self.learning_steps_reward = 0
            self.steps_exp = steps_exp

            # build synthetic noisy dataset
            self.noisy_dataset.build(
                actor=self.bc,
                noise_range=noise_range,
                n_trajs=n_demo_traj,
            )
            self.save_noisy_dataset = False

            # train reward function
            self.train_reward(size_reward_dataset)

    def train_reward(self, size_reward_dataset: int):
        """
        Train D-REX's reward function

        Parameters
        ----------
        size_reward_dataset: int
            size of the noise dataset
        """
        print("> Training reward function...")

        # train each reward function
        for i in range(len(self.reward_funcs)):
            print(f'Reward function: {i}')
            self.learning_steps_reward = 0

            # load data
            data = self.noisy_dataset.sample(size_reward_dataset)

            idxes = np.random.permutation(len(data))
            train_idxes = idxes[:int(len(data) * 0.8)]
            valid_idxes = idxes[int(len(data) * 0.8):]

            def _load(idx_list, add_noise=True):
                if len(idx_list) > self.batch_size_reward:
                    idx = np.random.choice(idx_list, self.batch_size_reward, replace=False)
                else:
                    idx = idx_list

                batch = []
                for j in idx:
                    batch.append(data[j])

                b_x, b_y, b_l = zip(*batch)
                x_split = np.array([len(x) for x in b_x], dtype=int)
                y_split = np.array([len(y) for y in b_y], dtype=int)
                # b_x, b_y, b_l = torch.cat(b_x, dim=0), torch.cat(b_y, axis=0), torch.tensor(b_l, device=self.device)
                b_x, b_y = torch.cat(b_x, dim=0), torch.cat(b_y, dim=0)
                b_l = np.array(b_l)

                if add_noise:
                    b_l = (b_l + np.random.binomial(1, 0.1, self.batch_size_reward)) % 2  # flip with probability 0.1

                return (
                    b_x,
                    b_y,
                    x_split,
                    y_split,
                    torch.tensor(b_l, dtype=torch.float, device=self.device),
                )

            for it in tqdm(range(self.iter_reward)):
                states_x, states_y, states_x_split, states_y_split, labels = _load(train_idxes, add_noise=True)
                logits_x = self.reward_funcs[i](states_x)
                logits_y = self.reward_funcs[i](states_y)

                ptr_x = 0
                logits_x_split = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                for i_split in range(states_x_split.shape[0]):
                    logits_x_split = \
                        torch.cat((logits_x_split, logits_x[ptr_x: ptr_x + states_x_split[i_split]].sum().unsqueeze(0)))
                    ptr_x += states_x_split[i_split]
                ptr_y = 0
                logits_y_split = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                for i_split in range(states_y_split.shape[0]):
                    logits_y_split = \
                        torch.cat((logits_y_split, logits_y[ptr_y: ptr_y + states_y_split[i_split]].sum().unsqueeze(0)))
                    ptr_y += states_y_split[i_split]

                labels = labels.long()
                logits_xy = torch.cat((logits_x_split.unsqueeze(1), logits_y_split.unsqueeze(1)), dim=1)
                loss_cal = torch.nn.CrossEntropyLoss()
                loss_cmp = loss_cal(logits_xy, labels)
                loss_l2 = self.l2_ratio_reward * parameters_to_vector(self.reward_funcs[i].parameters()).norm() ** 2
                loss_reward = loss_cmp + loss_l2
                self.reward_optims[i].zero_grad()
                loss_reward.backward()
                self.reward_optims[i].step()
                self.learning_steps_reward += 1

                if self.learning_steps_reward % 1000 == 0:
                    tqdm.write(f'step: {self.learning_steps_reward}, loss: {loss_reward.item():.3f}')
        print("> Done")

    def update(self):
        """
        Update the algorithm
        """
        self.learning_steps += 1
        states, actions, _, dones, log_pis, next_states = \
            self.buffer.get()
        rewards = torch.zeros((states.shape[0], 1)).type_as(states)
        with torch.no_grad():
            for i in range(len(self.reward_funcs)):
                rewards += self.reward_funcs[i](states) / len(self.reward_funcs)
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states)
