import torch

from typing import Optional

from .base import ControlAffineSystem, TrackingEnv
from .inverted_pendulum import InvertedPendulum
from .neural_lander import NeuralLander
from .cart_pole import CartPole
from .cart_double_pole import CartDoublePole
from .f16_gcas import F16GCAS
from .f16_tracking import F16Tracking


def make_env(
        env_id: str,
        device: torch.device = torch.device('cpu'),
) -> ControlAffineSystem:
    if env_id == 'InvertedPendulum':
        return InvertedPendulum(device)
    elif env_id == 'NeuralLander':
        return NeuralLander(device)
    elif env_id == 'CartPole':
        return CartPole(device)
    elif env_id == 'CartDoublePole':
        return CartDoublePole(device)
    elif env_id == 'F16GCAS':
        return F16GCAS(device)
    elif env_id == 'F16Tracking':
        return F16Tracking(device)
    else:
        raise NotImplementedError(f'{env_id} not implemented')

