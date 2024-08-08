#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    
    def init_hidden(self, batch_size, device=None):
        # Initialize hidden states for both actor and critic RNNs
        return (self.memory_a.init_hidden(batch_size, device), self.memory_c.init_hidden(batch_size, device))
    
    def get_hidden_out(self):
        # Return hidden states in a format that can be used by the RNNs
        return self.memory_a.hidden_out, self.memory_c.hidden_out


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
        
        self.type = type.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_out = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            if input.dim() < 3:
                input = input.unsqueeze(0)
            if hidden_states is not None:
                # Provided hidden states are not used for next step
                out, self.hidden_out = self.rnn(input, hidden_states)
            else:
                # inference mode (collection): use hidden states of last step
                out, self.hidden_states = self.rnn(input, self.hidden_states)
        return out

    def reset(self, dones=None):
        if dones is not None:
            dones = dones.bool()
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
    
    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # Initialize the hidden states for LSTM or GRU
        if self.type == "lstm":
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h_0, c_0)
        else:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return h_0
