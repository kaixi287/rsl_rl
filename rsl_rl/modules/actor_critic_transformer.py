#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticTransformer(ActorCritic):
    model_name = "transformer"

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        d_model = 256,
        d_ff = 1024,
        transformer_num_heads=4,
        transformer_num_layers=4,
        init_noise_std=1.0,
        observation_only=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        
        if not observation_only:
            num_actor_obs += num_actions
            num_critic_obs += num_actions

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_actor_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.memory_a = TransformerMemory(num_actor_obs, transformer_num_heads, transformer_num_layers, d_model, d_ff)
        self.memory_c = TransformerMemory(num_critic_obs, transformer_num_heads, transformer_num_layers, d_model, d_ff)

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")

    def act(self, observations, masks=None, reset_masks=None, **kwargs):
        input_a = self.memory_a(observations, masks, reset_masks)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, reset_masks=None, **kwargs):
        input_c = self.memory_c(critic_observations, masks, reset_masks)
        return super().evaluate(input_c.squeeze(0))


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float, max_len: int = 100) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a vector of shape (max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:x.shape[0]])
        return self.dropout(x)
    
class ExtendedEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, activation=nn.ReLU(), intermediate_dim=None):
        super(ExtendedEmbedding, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = d_model  # Optionally set the intermediate dimension

        # First linear layer maps from input_dim to intermediate_dim
        self.linear1 = nn.Linear(input_dim, intermediate_dim)
        
        # Activation function, e.g., ReLU
        self.activation = activation
        
        # Second linear layer maps from intermediate_dim to d_model
        self.linear2 = nn.Linear(intermediate_dim, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerMemory(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()

        # self.embedding = nn.Linear(input_dim, d_model)
        # self.embedding = ExtendedEmbedding(input_dim, d_model, activation, d_embed)
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        transformer_layer =nn.TransformerEncoderLayer(d_model=input_dim,
                                                      nhead=num_heads,
                                                      dim_feedforward=d_ff,
                                                      dropout=dropout
                                                      )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        # self.initialize_transformer()
    
    def initialize_transformer(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:  # Applies to weights of linear layers and not biases
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, masks=None, reset_masks=None):

        # if masks is None:
        #     x = x[-1].unsqueeze(0)
        
        seq_len = x.size(0)

        if reset_masks is not None:
            # mask should be (batch_size, seq_len), with True values for positions to ignore
            reset_masks = torch.where(reset_masks.t() == 0, torch.tensor(float('-inf'), device=x.device), torch.tensor(0.0, device=x.device))

        # Generate a causal mask to limit attention to the preceding tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Embed the input (seq_len, batch_size, num_obs) --> (seq_len, batch_size, d_model)
        # x = self.embedding(x)
        x = self.pos_encoder(x) # (seq_len, batch_size, d_model)

        # Pass through the transformer.
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=reset_masks)   # (seq_len, batch_size, d_model)
        
        if masks is not None:
            x = unpad_trajectories(x, masks)
        else:
            x = x[-1]   # take only the last output in inference mode
            # print(f"output x shape in inference mode: {x.shape}")   #(4096, 256)

        return x
