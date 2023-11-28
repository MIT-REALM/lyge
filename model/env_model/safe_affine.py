import torch
import torch.nn as nn

from .affine import NeuralAffineEnv

from model.env.base import ControlAffineSystem
from model.buffer import DemoBuffer


class NeuralSafeCriticalAffineEnv(NeuralAffineEnv):
    """
    Safe critical affine model. States around the safe buffer in the
        relaxation area will be considered safe.
    """

    def __init__(
            self,
            base_env: ControlAffineSystem,  # to simulate this env
            device: torch.device,
            safe_buffer: DemoBuffer,
            normalize: bool = False,
            sparse_g: bool = False,
            hidden_layers: tuple = (128, 128, 128),
            hidden_activation: nn.Module = nn.Tanh(),
            goal_relaxation: float = 0.1
    ):
        states, actions, next_states = safe_buffer.all_transitions
        xdots = (next_states - states) / base_env.dt

        super().__init__(
            base_env=base_env,
            device=device,
            xdot_mean=torch.mean(xdots, dim=0),
            state_std=torch.std(states, dim=0),
            ctrl_std=torch.std(actions, dim=0),
            xdot_std=torch.std(xdots, dim=0),
            normalize=normalize,
            sparse_g=sparse_g,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            goal_relaxation=goal_relaxation,
        )

        self.safe_buffer = safe_buffer

    def sample_safe_states(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_states(batch_size)

    def sample_safe_ctrls(self, batch_size: int) -> torch.Tensor:
        return self.safe_buffer.sample_policy(batch_size)[1]
