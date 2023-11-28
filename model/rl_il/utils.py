import torch
import torch.nn as nn
import numpy as np
import pickle

from typing import Tuple, List

from ..env.base import ControlAffineSystem


def calculate_gae(
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        gamma: float,
        lambd: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate generalized advantage estimator

    Parameters
    ----------
    values: torch.Tensor
        values of the states
    rewards: torch.Tensor
        rewards given by the reward function
    dones: torch.Tensor
        if this state is the end of the episode
    next_values: torch.Tensor
        values of the next states
    gamma: float
        discount factor
    lambd: float
        lambd factor

    Returns
    -------
    advantages: torch.Tensor
        advantages
    gaes: torch.Tensor
        normalized gae
    """
    # calculate TD errors
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # initialize gae
    gaes = torch.empty_like(rewards)

    # calculate gae recursively from behind
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.shape[0] - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std(dim=0) + 1e-8)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update for SAC"""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network: nn.Module):
    """Disable the gradients of parameters in the network"""
    for param in network.parameters():
        param.requires_grad = False


class NoisePreferenceDataset:
    """
    Synthetic dataset by injecting noise in the bc policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to collect data
    device: torch.device
        cpu or cuda
    max_steps: int
        maximum steps in a slice
    min_margin: float
        minimum margin between two samples
    """
    def __init__(
            self,
            env: ControlAffineSystem,
            device: torch.device,
            max_steps: int = None,
            min_margin: float = None
    ):
        self.env = env
        self.device = device
        self.max_steps = max_steps
        self.min_margin = min_margin
        self.trajs = []

    def get_noisy_traj(self, actor: nn.Module, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one noisy trajectory

        Parameters
        ----------
        actor: nn.Module
            policy network
        noise_level: float
            noise to inject

        Returns
        -------
        states: torch.Tensor
            states in the noisy trajectory
        action: torch.Tensor
            actions in the noisy trajectory
        rewards: torch.Tensor
            rewards of the s-a pairs
        """
        states, actions, rewards = [], [], []

        state = self.env.reset()
        t = 0
        while True:
            t += 1
            if np.random.rand() < noise_level:
                action = self.env.sample_ctrls(1)
            else:
                action = actor(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state.unsqueeze(0))
            actions.append(action)
            rewards.append(reward)
            if done or t >= self.env.max_episode_steps:
                break
            state = next_state

        return (
            torch.cat(states, dim=0),
            torch.cat(actions, dim=0),
            torch.tensor(rewards, device=self.device)
        )

    def build(self, actor: nn.Module, noise_range: np.array, n_trajs: int):
        """
        Build noisy dataset

        Parameters
        ----------
        actor: nn.Module
            policy network
        noise_range: np.array
            range of noise
        n_trajs: int
             number of trajectories
        """
        print('> Collecting noisy demonstrations')
        for noise_level in noise_range:
            agent_trajs = []
            reward_traj = 0
            for i_traj in range(n_trajs):
                states, actions, rewards = self.get_noisy_traj(actor, noise_level)
                agent_trajs.append((states, actions, rewards))
                reward_traj += rewards.sum()
            self.trajs.append((noise_level, agent_trajs))
            reward_traj /= n_trajs
            print(f'Noise level: {noise_level:.3f}, traj reward: {reward_traj:.3f}')
        print('> Done')

    def sample(self, n_sample: int) -> List:
        """
        Sample from the data set

        Parameters
        ----------
        n_sample: int
            number of samples

        Returns
        -------
        data: List
            list of trajectories, each element contains:
                1.) trajectory 1
                2.) trajectory 2
                3.) whether trajectory 1's reward is larger than trajectory 2
        """
        data = []

        for _ in range(n_sample):
            # pick two noise level set
            x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

            # pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # sub-sampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0]) - self.max_steps)
                x_slice = slice(ptr, ptr + self.max_steps)
            else:
                x_slice = slice(len(x_traj[0]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0]) - self.max_steps)
                y_slice = slice(ptr, ptr + self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # done
            data.append(
                (x_traj[0][x_slice],
                 y_traj[0][y_slice],
                 0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
            )

        return data

    def save(self, save_dir: str):
        """
        Save the dataset
        Parameters
        ----------
        save_dir: str
            path to save
        """
        with open(f'{save_dir}/noisy_trajs.pkl', 'wb') as f:
            pickle.dump(self.trajs, f)


