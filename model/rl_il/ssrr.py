import torch
import torch.nn as nn
import numpy as np
import scipy
import pickle

from scipy.optimize import curve_fit
from typing import Tuple
from torch.optim import Adam
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector

from .ppo import PPO
from .airl import AIRL
from ..buffer import DemoBuffer
from ..env.base import ControlAffineSystem
from ..network.mlp import NormalizedMLP


class NoisePreferenceBuffer:
    """
    synthetic dataset by injecting noise in the actor, used in SSRR
    """
    def __init__(
            self,
            env: ControlAffineSystem,
            airl: AIRL,
            device: torch.device,
            max_steps: int = 100,
            min_margin: float = 0
    ):
        self.env = env
        self.actor = airl.actor
        self.device = device
        self.reward_func = airl.disc.get_reward
        self.max_steps = max_steps
        self.min_margin = min_margin
        self.trajs = []
        self.noise_reward = None

    def get_noisy_traj(self, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one noisy trajectory
        Parameters
        ----------
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
        next_states: torch.Tensor
            next states the agent transferred to
        """
        states, actions, rewards, next_states = [], [], [], []

        state = self.env.reset()
        t = 0
        while True:
            t += 1
            if np.random.rand() < noise_level:
                action = self.env.sample_ctrls(1)
            else:
                action = self.actor(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state.unsqueeze(0))
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.unsqueeze(0))
            if done or t >= self.env.max_episode_steps:
                break
            state = next_state

        return (
            torch.cat(states, dim=0),
            torch.cat(actions, dim=0),
            torch.tensor(rewards, dtype=torch.float, device=self.device),
            torch.cat(next_states, dim=0),
        )

    def build(self, noise_range: np.array, n_trajs: int):
        """
        Build noisy buffer
        Parameters
        ----------
        noise_range: np.array
            range of noise
        n_trajs: int
             number of trajectories
        """
        print('Collecting noisy demonstrations')
        for noise_level in noise_range:
            agent_trajs = []
            reward_traj = 0
            for i_traj in range(n_trajs):
                states, actions, rewards, next_states = self.get_noisy_traj(noise_level)
                reward_traj += rewards.sum()

                # if given reward function, use that instead of the ground truth
                if self.reward_func is not None:
                    rewards = self.reward_func(states)

                agent_trajs.append((states, actions, rewards, next_states))
            self.trajs.append((noise_level, agent_trajs))
            reward_traj /= n_trajs
            print(f'Noise level: {noise_level:.3f}, traj reward: {reward_traj:.3f}')
        print('Collecting finished')

    def get(self) -> list:
        """
        Get all the trajectories
        Returns
        -------
        trajs: list, all trajectories
        """
        return self.trajs

    def get_noise_reward(self):
        """
        Get rewards and the corresponding noise level
        Returns
        -------
        noise_reward: list
            each element is an array with (noise_level, reward)
        """
        if self.noise_reward is None:
            self.noise_reward = []

            prev_noise = 0.0
            noise_reward = 0
            n_traj = 0
            for traj in self.trajs:
                for agent_traj in traj[1]:
                    if prev_noise == traj[0]:  # noise level has not changed
                        noise_reward += agent_traj[2].mean()
                        n_traj += 1
                        prev_noise = traj[0]
                    else:  # noise level changed
                        self.noise_reward.append([prev_noise, noise_reward.cpu().detach().numpy() / n_traj])
                        prev_noise = traj[0]
                        noise_reward = agent_traj[2].mean()
                        n_traj = 1
            self.noise_reward = np.array(self.noise_reward, dtype=np.float32)
        return self.noise_reward

    def sample(self, n_sample: int):
        """
        Sample data from the buffer
        Parameters
        ----------
        n_sample: int
            number of samples
        Returns
        -------
        data: list
            each element contains (noise level, (states, actions, rewards) in the trajectory)
        """
        data = []
        for _ in range(n_sample):
            noise_idx = np.random.choice(len(self.trajs))
            traj = self.trajs[noise_idx][1][np.random.choice(len(self.trajs[noise_idx][1]))]
            if len(traj[0]) > self.max_steps:
                ptr = np.random.randint(len(traj[0]) - self.max_steps)
                x_slice = slice(ptr, ptr + self.max_steps)
            else:
                x_slice = slice(len(traj[0]))
            states, actions, rewards, _ = traj
            data.append((self.trajs[noise_idx][0], (states[x_slice], actions[x_slice], rewards[x_slice])))
        return data

    def save(self, save_dir: str):
        """
        Save the buffer
        Parameters
        ----------
        save_dir: str
            path to save
        Returns
        -------
        """
        with open(f'{save_dir}/noisy_trajs.pkl', 'wb') as f:
            pickle.dump(self.trajs, f)


class SSRR(PPO):
    """
    Implementation of SSRR

    Reference:
    [1] Chen, L., Paleja, R., and Gombolay, M.
    Learning from suboptimal demonstration via self-supervised reward regression.
    In Conference on Robot Learning. PMLR, 2020.
    """
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
            airl_path: str = None,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            batch_size_reward: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_reward: float = 1e-4,
            units_actor: tuple = (128, 128),
            units_critic: tuple = (128, 128),
            units_reward: tuple = (128, 128),
            epoch_ppo: int = 10,
            iter_reward: int = 10000,
            l2_ratio_reward: float = 0.01,
            n_reward_model: int = 3,
            noise_range: np.array = np.arange(0., 1., 0.05),
            n_demo_traj: int = 5,
            steps_exp: int = 50,
            clip_eps: float = 0.2,
            lambd: float = 0.95,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10
    ):
        super().__init__(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                         gamma, rollout_length, mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
                         epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # expert's buffer
        self.demo_buffer = demo_buffer

        # use the given AIRL model
        if airl_path is not None:
            self.airl = AIRL(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std)
            self.airl.load(airl_path, device)
            print('AIRL loaded')

            # noisy buffer
            self.noisy_buffer = NoisePreferenceBuffer(
                env=env,
                airl=self.airl,
                device=self.device,
                max_steps=steps_exp,
            )
            self.env = env
            self.noisy_buffer.build(noise_range=noise_range, n_trajs=n_demo_traj)
            print('Noisy buffer built')
            self.save_noisy_buffer = False

            # fit noise-performance relationship using sigmoid function
            self.sigma = None
            self.fit_noise_performance()

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
            self.save_reward_func = False
            self.steps_exp = steps_exp
            self.loss_reward = nn.MSELoss()
            self.train_reward()

    def fit_noise_performance(self):
        """Fit the noise-performance curve"""
        print('Fitting noise-performance')

        # sigmoid curve
        def sigmoid(eff, x):
            x0, y0, c, k = eff
            y = c / (1 + np.exp(-k * (x - x0))) + y0
            return y

        def residuals(eff, x, y):
            return y - sigmoid(eff, x)

        # fit the curve
        noise_reward_data = self.noisy_buffer.get_noise_reward()
        label = noise_reward_data[:, 1]
        label_scale = label.max() - label.min()
        label_intercept = label.min()
        label = (label - label_intercept) / label_scale
        noises = noise_reward_data[:, 0]

        eff_guess = np.array([np.median(noises), np.median(label), 1.0, -1.0])
        eff_fit, cov, infodict, mesg, ier = scipy.optimize.leastsq(
            residuals, eff_guess, args=(noises, label), full_output=True)

        def fitted_sigma(x):
            return sigmoid(eff_fit, x) * label_scale + label_intercept

        self.sigma = fitted_sigma

    def train_reward(self):
        """Train SSRR's reward function"""
        print('Training reward function')
        for i in range(len(self.reward_funcs)):
            print(f'Reward function: {i}')
            self.learning_steps_reward = 0
            for it in tqdm(range(self.iter_reward), ncols=80):
                self.learning_steps_reward += 1

                # load data
                data = self.noisy_buffer.sample(self.batch_size_reward)
                noises, trajs = zip(*data)
                noises = np.asarray(noises)
                states, actions, _ = zip(*trajs)
                lengths = [x.shape[0] for x in states]
                states = torch.cat([x for x in states], dim=0).detach()
                rewards = self.reward_funcs[i](states).squeeze(0)

                # calculate traj rewards, using AIRL's reward function
                traj_rewards = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                ptr = 0
                for length in lengths:
                    traj_rewards = torch.cat((traj_rewards, rewards[ptr: ptr + length].sum().unsqueeze(0)))
                    ptr += length

                # target reward is sigma(eta)
                targets = torch.tensor(self.sigma(noises) * np.array(lengths) / self.env.max_episode_steps,
                                       dtype=torch.float, device=self.device)

                # update
                loss_cmp = self.loss_reward(traj_rewards, targets)
                loss_l2 = self.l2_ratio_reward * parameters_to_vector(self.reward_funcs[i].parameters()).norm() ** 2
                loss = loss_cmp + loss_l2
                self.reward_optims[i].zero_grad()
                loss.backward()
                self.reward_optims[i].step()

                if self.learning_steps_reward % 1000 == 0:
                    tqdm.write(f'step: {self.learning_steps_reward}, loss: {loss.item():.3f}')
        print("Reward function finished training")
