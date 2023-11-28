import torch
import numpy as np
import os
import pickle

from typing import Tuple


class TrackingBuffer:

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        self._n = 0  # current num of demos
        self._p = 0  # pointer
        self.buffer_size = buffer_size
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # record reference path
        self._states_ref = torch.empty(
            (buffer_size, state_dim), dtype=torch.float, device=device)
        self._actions_ref = torch.empty(
            (buffer_size, action_dim), dtype=torch.float, device=device)

        # record demonstrations
        self._states = torch.empty(
            (buffer_size, state_dim), dtype=torch.float, device=device)
        self._actions = torch.empty(
            (buffer_size, action_dim), dtype=torch.float, device=device)
        self._rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self._dones = torch.empty(
            (buffer_size, 1), dtype=torch.int, device=device)
        self._next_states = torch.empty(
            (buffer_size, state_dim), dtype=torch.float, device=device)

    def append(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: float,
            done: bool,
            next_state: torch.Tensor,
            state_ref: torch.Tensor,
            action_ref: torch.Tensor
    ):
        if state.ndim == 2:
            state = state.squeeze(0)
        if action.ndim == 2:
            action = action.squeeze(0)
        if next_state.ndim == 2:
            next_state = next_state.squeeze(0)
        if state_ref.ndim == 2:
            state_ref = state_ref.squeeze(0)
        if action_ref.ndim == 2:
            action_ref = action_ref.squeeze(0)

        self._states[self._p].copy_(state)
        self._actions[self._p].copy_(action)
        self._rewards[self._p] = float(reward)
        self._dones[self._p] = bool(done)
        self._next_states[self._p].copy_(next_state)
        self._states_ref[self._p].copy_(state_ref)
        self._actions_ref[self._p].copy_(action_ref)

        self._p += 1
        if self._p > self.buffer_size:
            raise AssertionError('Demo buffer is full')

        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'states': self._states.clone().cpu(),
            'actions': self._actions.clone().cpu(),
            'rewards': self._rewards.clone().cpu(),
            'dones': self._dones.clone().cpu(),
            'next_states': self._next_states.clone().cpu(),
            'states_ref': self._states_ref.clone().cpu(),
            'actions_ref': self._actions_ref.clone().cpu()
        }, path)

    def load(self, path: str):
        data = torch.load(path)

        # disable grads
        for key in data.keys():
            data[key].requires_grad = False

        self._states.copy_(data['states'])
        self._actions.copy_(data['actions'])
        self._rewards.copy_(data['rewards'])
        self._dones.copy_(data['dones'])
        self._next_states.copy_(data['next_states'])
        self._states_ref.copy_(data['states_ref'])
        self._actions_ref.copy_(data['actions_ref'])
        self._n = data['states'].shape[0]
        self._p = data['states'].shape[0]

    def clear(self):
        self._n = 0
        self._p = 0

    def sample_policy(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return (
            self._states[idxes],
            self._actions[idxes],
            self._states_ref[idxes],
            self._actions_ref[idxes]
        )

    def expand(self, buffer_size):
        pre_buffer_size = self.buffer_size
        tmp_states, tmp_actions = self._states.clone(), self._actions.clone()
        tmp_rewards, tmp_dones = self._rewards.clone(), self._dones.clone()
        tmp_next_states = self._next_states.clone()
        tmp_states_ref, tmp_actions_ref = self._states_ref.clone(), self._actions_ref.clone()
        self._states = torch.empty(
            (buffer_size, self.state_dim), dtype=torch.float, device=self.device)
        self._actions = torch.empty(
            (buffer_size, self.action_dim), dtype=torch.float, device=self.device)
        self._rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=self.device)
        self._dones = torch.empty(
            (buffer_size, 1), dtype=torch.int, device=self.device)
        self._next_states = torch.empty(
            (buffer_size, self.state_dim), dtype=torch.float, device=self.device)
        self._states_ref = torch.empty(
            (buffer_size, self.state_dim), dtype=torch.float, device=self.device)
        self._actions_ref = torch.empty(
            (buffer_size, self.action_dim), dtype=torch.float, device=self.device)
        self._states[:pre_buffer_size, :].copy_(tmp_states)
        self._actions[:pre_buffer_size, :].copy_(tmp_actions)
        self._rewards[:pre_buffer_size, :].copy_(tmp_rewards)
        self._dones[:pre_buffer_size, :].copy_(tmp_dones)
        self._next_states[:pre_buffer_size, :].copy_(tmp_next_states)
        self._states_ref[:pre_buffer_size, :].copy_(tmp_states_ref)
        self._actions_ref[:pre_buffer_size, :].copy_(tmp_actions_ref)
        self.buffer_size = buffer_size

    def sample_states(self, batch_size: int) -> torch.Tensor:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return self._states[idxes]

    def sample_states_ref(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return self._states[idxes], self._states_ref[idxes]

    def sample_transition(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return (
            self._states[idxes],
            self._actions[idxes],
            self._next_states[idxes]
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 2, size=batch_size)
        return (
            torch.cat((self._states[idxes], self._states_ref[idxes], self._actions_ref[idxes]), dim=1),
            self._actions[idxes],
            self._rewards[idxes],
            self._dones[idxes],
            torch.cat((self._next_states[idxes], self._states_ref[idxes + 1], self._actions_ref[idxes + 1]), dim=1)
        )

    @property
    def mean_reward(self):
        return torch.sum(self._rewards) / torch.sum(self._dones)

    @property
    def state_std(self):
        return torch.std(torch.cat((self._states, self._states_ref), dim=0), dim=0)

    @property
    def action_std(self):
        return torch.std(torch.cat((self._actions, self._actions_ref), dim=0), dim=0)

    @property
    def all_transitions(self):
        return self._states, self._actions, self._next_states

    @property
    def all_data(self):
        return (
            self._states,
            self._actions,
            self._rewards,
            self._dones,
            self._next_states,
            self._states_ref,
            self._actions_ref
        )


class DemoBuffer:

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        self._n = 0  # current num of demos
        self._p = 0  # pointer
        self.buffer_size = buffer_size
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._states = torch.empty(
            (buffer_size, state_dim), dtype=torch.float, device=device)
        self._actions = torch.empty(
            (buffer_size, action_dim), dtype=torch.float, device=device)
        self._rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self._dones = torch.empty(
            (buffer_size, 1), dtype=torch.int, device=device)
        self._next_states = torch.empty(
            (buffer_size, state_dim), dtype=torch.float, device=device)

    def append(self, state: torch.Tensor, action: torch.Tensor, reward: float, done: bool, next_state: torch.Tensor):
        if state.ndim == 2:
            state = state.squeeze(0)
        if action.ndim == 2:
            action = action.squeeze(0)
        if next_state.ndim == 2:
            next_state = next_state.squeeze(0)

        self._states[self._p].copy_(state)
        self._actions[self._p].copy_(action)
        self._rewards[self._p] = float(reward)
        self._dones[self._p] = bool(done)
        self._next_states[self._p].copy_(next_state)

        self._p += 1
        if self._p > self.buffer_size:
            raise AssertionError('Demo buffer is full')

        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'states': self._states.clone().cpu(),
            'actions': self._actions.clone().cpu(),
            'rewards': self._rewards.clone().cpu(),
            'dones': self._dones.clone().cpu(),
            'next_states': self._next_states.clone().cpu(),
        }, path)

    def load(self, path: str):
        data = torch.load(path)

        # disable grads
        for key in data.keys():
            data[key].requires_grad = False

        self._states.copy_(data['states'])
        self._actions.copy_(data['actions'])
        self._rewards.copy_(data['rewards'])
        self._dones.copy_(data['dones'])
        self._next_states.copy_(data['next_states'])
        self._n = data['states'].shape[0]
        self._p = data['states'].shape[0]

    def clear(self):
        self._n = 0
        self._p = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return (
            self._states[idxes],
            self._actions[idxes],
            self._rewards[idxes],
            self._dones[idxes],
            self._next_states[idxes]
        )

    def sample_transition(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return (
            self._states[idxes],
            self._actions[idxes],
            self._next_states[idxes]
        )

    def sample_policy(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return (
            self._states[idxes],
            self._actions[idxes],
        )

    def sample_states(self, batch_size: int) -> torch.Tensor:
        idxes = np.random.randint(low=0, high=self._n - 1, size=batch_size)
        return self._states[idxes]

    def expand(self, buffer_size):
        pre_buffer_size = self.buffer_size
        tmp_states, tmp_actions = self._states.clone(), self._actions.clone()
        tmp_rewards, tmp_dones = self._rewards.clone(), self._dones.clone()
        tmp_next_states = self._next_states.clone()
        self._states = torch.empty(
            (buffer_size, self.state_dim), dtype=torch.float, device=self.device)
        self._actions = torch.empty(
            (buffer_size, self.action_dim), dtype=torch.float, device=self.device)
        self._rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=self.device)
        self._dones = torch.empty(
            (buffer_size, 1), dtype=torch.int, device=self.device)
        self._next_states = torch.empty(
            (buffer_size, self.state_dim), dtype=torch.float, device=self.device)
        self._states[:pre_buffer_size, :].copy_(tmp_states)
        self._actions[:pre_buffer_size, :].copy_(tmp_actions)
        self._rewards[:pre_buffer_size, :].copy_(tmp_rewards)
        self._dones[:pre_buffer_size, :].copy_(tmp_dones)
        self._next_states[:pre_buffer_size, :].copy_(tmp_next_states)
        self.buffer_size = buffer_size

    @property
    def mean_reward(self):
        return torch.sum(self._rewards) / torch.sum(self._dones)

    @property
    def all_states(self):
        return self._states

    @property
    def all_transitions(self):
        return self._states, self._actions, self._next_states

    @property
    def all_data(self):
        return self._states, self._actions, self._rewards, self._dones, self._next_states

    @property
    def state_std(self):
        return torch.std(self._states, dim=0)

    @property
    def action_std(self):
        return torch.std(self._actions, dim=0)


class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents
    """

    def __init__(
            self,
            buffer_size: int,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            mix: int = 1
    ):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.device = device
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, state_dim), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, action_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, state_dim), dtype=torch.float, device=device)

    def append(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: float,
            done: bool,
            log_pi: float,
            next_state: torch.Tensor
    ):
        """
        Save a transition in the buffer
        """
        if state.ndim == 2:
            state = state.squeeze(0)
        if action.ndim == 2:
            action = action.squeeze(0)

        self.states[self._p].copy_(state)
        self.actions[self._p].copy_(action)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(next_state)

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data in the buffer

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
