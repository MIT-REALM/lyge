import torch
import torch.nn as nn

from typing import Tuple

from .base import Controller
from .neural_controller import NeuralController

from model.network.pd_quadratic import PDQuadraticNet


class NeuralCLFController(Controller):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            hidden_layers_lyapunov: tuple = (128, 128),
            hidden_layers_controller: tuple = (128, 128),
            clf_lambda: float = 1.0
    ):
        super(NeuralCLFController, self).__init__(goal_point, u_eq, state_std, ctrl_std)

        # set up controller
        self.controller = NeuralController(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_point=goal_point,
            u_eq=u_eq,
            state_std=state_std,
            ctrl_std=ctrl_std,
            hidden_layers=hidden_layers_controller
        )

        # set up Lyapunov network
        # self.lyapunov = QuadraticMLP(
        #     in_dim=state_dim,
        #     hidden_layers=hidden_layers_lyapunov,
        #     hidden_activation=nn.Tanh(),
        # )
        self.lyapunov = PDQuadraticNet(
            in_dim=state_dim,
            hidden_layers=hidden_layers_lyapunov,
            hidden_activation=nn.Tanh(),
        )

        # save parameters
        self.clf_lambda = clf_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.controller(x)

    def set_controller(self, controller: NeuralController):
        self.controller.load_state_dict(controller.state_dict())

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """get the control input for a given state with a nominal controller"""
        return self.controller(x)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return self.controller.act(x)

    def lyapunov_lie_derivatives(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the lie derivatives
            L_f = (\partial v / \partial x) * f(x)
            L_g = (\partial v / \partial x) * g(x)
        """
        # # get the Jacobian of V for each entry in the batch
        # _, grad_v = self.lyapunov.forward_jacobian(x)
        #
        # # multiply f, g with the Jacobian to get the Lie derivatives
        # lf = torch.bmm(grad_v, f).squeeze(1)
        # lg = torch.bmm(grad_v, g).squeeze(1)
        #
        # return lf, lg
        raise NotImplementedError

    def V_with_Jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_trans = self.normalize_state(x)
        V, JV = self.lyapunov.forward_jacobian(x_trans)
        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = self.normalize_state(x)
        return self.lyapunov(x_trans)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def disable_grad_lyapunov(self):
        self.lyapunov.disable_grad()

    def disable_grad_ctrl(self):
        self.controller.disable_grad()
