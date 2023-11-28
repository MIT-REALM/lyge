import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from .ppo import PPO
from ..buffer import DemoBuffer
from ..network.disc import AIRLDiscrim


class AIRL(PPO):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            demo_buffer: DemoBuffer = None,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            batch_size: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_disc: float = 3e-4,
            units_actor: tuple = (128, 128),
            units_critic: tuple = (128, 128),
            units_disc_r: tuple = (128, 128),
            units_disc_v: tuple = (128, 128),
            epoch_ppo: int = 50,
            epoch_disc: int = 10,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0
    ):
        super().__init__(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                         gamma, rollout_length, mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
                         epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # expert's buffer
        self.demo_buffer = demo_buffer

        # discriminator
        self.disc = AIRLDiscrim(
            state_dim=state_dim,
            goal_point=goal_point,
            state_std=state_std,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # samples from current policy's trajectories
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)

            # samples from expert's demonstrations
            states_exp, actions_exp, _, dones_exp, next_states_exp = self.demo_buffer.sample(self.batch_size)

            # calculate log probabilities of expert actions
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # update discriminator
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp
            )

        # we don't use reward signals here
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # calculate rewards
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)

        # update PPO using estimated rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states)

    def update_disc(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor,
            states_exp: torch.Tensor,
            dones_exp: torch.Tensor,
            log_pis_exp: torch.Tensor,
            next_states_exp: torch.Tensor,
    ):
        # output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'disc': self.disc.state_dict()
        }, path)

    def load(self, path: str, device: torch.device):
        data = torch.load(path, map_location=device)
        self.actor.load_state_dict(data['actor'], strict=False)
        self.disc.load_state_dict(data['disc'], strict=False)
