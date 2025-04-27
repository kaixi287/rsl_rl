from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class ActorCriticHistory(nn.Module):
    is_recurrent = False
    with_history = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        history_len=48,  # Number of observations in history
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticHistory.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Updated input dimensions: concatenate H observations
        mlp_input_dim_a = num_actor_obs * history_len
        mlp_input_dim_c = num_critic_obs * history_len

        # Actor MLP
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation]
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers += [
                nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]),
                activation
            ]
        actor_layers += [nn.Linear(actor_hidden_dims[-1], num_actions)]
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation]
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers += [
                nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]),
                activation
            ]
        critic_layers += [nn.Linear(critic_hidden_dims[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP with History: {self.actor}")
        print(f"Critic MLP with History: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, stacked_observations):
        mean = self.actor(stacked_observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, stacked_observations, **kwargs):
        self.update_distribution(stacked_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, stacked_observations, **kwargs):
        return self.actor(stacked_observations)

    def evaluate(self, stacked_critic_observations, **kwargs):
        return self.critic(stacked_critic_observations)
