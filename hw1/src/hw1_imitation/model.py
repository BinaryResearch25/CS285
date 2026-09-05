"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        output_dim = chunk_size * action_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.net(state)

        pred = pred.reshape(
            -1,
            self.chunk_size,
            self.action_dim,
        )

        loss = torch.mean((pred - action_chunk) ** 2)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        pred = self.net(state)

        pred = pred.view(
            -1,
            self.chunk_size,
            self.action_dim,
        )

        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        action_chunk_dim = chunk_size * action_dim

        input_dim = (
            state_dim 
            + action_chunk_dim
            + 1
        )

        output_dim = action_chunk_dim

        layers = []

        for hidden_dim in hidden_dims:
            layers.append(
                nn.Linear(input_dim, hidden_dim)
            )
            layers.append(nn.ReLU())

            input_dim = hidden_dim

        layers.append(
            nn.Linear(input_dim, output_dim)
        )

        self.net = nn.Sequential(*layers)    

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = state.shape[0]

        noise = torch.randn_like(action_chunk)

        tau = torch.rand(batch_size, 1, 1, device=state.device)

        action_tau = (
            tau * action_chunk 
            + (1 - tau) * noise
        )

        target_velocity = (
            action_chunk - noise
        )

        pred_velocity = self._pred_velocity(
            state=state,
            action=action_tau,
            tau=tau,
        )

        loss = torch.mean(
            (pred_velocity 
             - target_velocity) ** 2
        )

        return loss


 
    def _pred_velocity(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        #velocity = action_chunk - noise
        batch_size = state.shape[0]

        action_flat = action.reshape(
            batch_size,
            -1,
        )

        tau_flat = tau.reshape(
            batch_size,
            1,
        )

        x = torch.cat(
            [
                state, 
                action_flat, 
                tau_flat,
            ],
            dim=-1,
        )

        velocity = self.net(x)
        velocity = velocity.reshape(
            batch_size,
            self.chunk_size,
            self.action_dim,
        )

        return velocity

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        batch_size = state.shape[0]

        action = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
        )


        dt = 1.0 / num_steps

        for i in range(num_steps):
            tau_value = (i + 0.5) / num_steps

            tau = torch.full(
                (batch_size, 1, 1),
                fill_value=tau_value,
                device=state.device,
            )

            velocity = (
                self._pred_velocity(
                    state=state,
                    action=action,
                    tau=tau,
                )
            )

            action = (
                action 
                + dt * velocity
            )

        return action
        

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
