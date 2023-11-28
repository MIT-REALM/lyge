import torch

from .base import Agent
from .ppo import PPO
from .airl import AIRL
from .drex import DREX
from .ssrr import SSRR


AGENTS = {
    'ppo',
    'airl',
    'drex',
    'ssrr'
}


def get_agent(
        algo: str,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        goal_point: torch.Tensor,
        u_eq: torch.Tensor,
        state_std: torch.Tensor,
        ctrl_std: torch.Tensor,
        kwargs: dict = None
) -> Agent:
    if algo == 'ppo':
        return PPO(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std)
    elif algo == 'airl':
        if kwargs is not None:
            return AIRL(
                state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                kwargs['demo_buffer']
            )
        else:
            return AIRL(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std)
    elif algo == 'drex':
        if kwargs is not None:
            return DREX(
                state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                kwargs['env'], kwargs['demo_buffer'], kwargs['policy_bc']
            )
        else:
            return DREX(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std)
    elif algo == 'ssrr':
        if kwargs is not None:
            return SSRR(
                state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std,
                kwargs['env'], kwargs['demo_buffer'], kwargs['airl_path']
            )
        else:
            return SSRR(state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std)
    else:
        raise NotImplementedError(f'{algo} has not been implemented')
