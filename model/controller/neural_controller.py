import torch
import torch.nn as nn

from .base import Controller

from model.network.mlp import NormalizedMLP


class NeuralController(Controller):
    """
    Neural network controller.

    Parameters
    ----------
    state_dim: int
    action_dim: int
    hidden_layers: tuple
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            hidden_layers: tuple = (128, 128)
    ):
        super(NeuralController, self).__init__(goal_point, u_eq, state_std, ctrl_std)
        self.net = NormalizedMLP(
            in_dim=state_dim,
            out_dim=action_dim,
            input_mean=goal_point.squeeze(),
            input_std=state_std,
            hidden_layers=hidden_layers,
            hidden_activation=nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        # x_trans = self.normalize_state(x)
        u_trans = self.net(x)
        return self.de_normalize_action(u_trans)
        # return self.net(x)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        # x_trans = self.normalize_state(x)
        # with torch.no_grad():
        #     u_trans = self.net(x_trans)
        # return self.de_normalize_action(u_trans)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            # return self.net(x)
            u_trans = self.net(x)
            return self.de_normalize_action(u_trans)
