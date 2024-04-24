#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticTransformer(ActorCritic):
    method = "transformer"

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        max_seq_len = 24,
        d_model = 512,
        transformer_num_heads=8,
        transformer_num_layers=6,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=d_model,
            num_critic_obs=d_model,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.memory_a = TransformerMemory(num_actor_obs, max_seq_len, transformer_num_heads, transformer_num_layers, d_model)
        self.memory_c = TransformerMemory(num_critic_obs, max_seq_len, transformer_num_heads, transformer_num_layers, d_model)

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")

    def act(self, observations, masks=None, **kwargs):
        input_a = self.memory_a(observations, masks)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, **kwargs):
        input_c = self.memory_c(critic_observations, masks)
        return super().evaluate(input_c.squeeze(0))


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1) # (seq_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:x.shape[1], :, :]).requires_grad_(False)
        return self.dropout(x)

class TransformerMemory(nn.Module):
    def __init__(self, input_dim, max_seq_len, num_heads, num_layers, d_model, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        transformer_layer =nn.TransformerEncoderLayer(d_model=d_model,
                                                      nhead=num_heads,
                                                      dim_feedforward=d_ff,
                                                      dropout=dropout
                                                      )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
    
    def forward(self, x, masks=None):

        seq_len = x.size(0)

        if masks is not None:
            # Training mode
            # Padding mask should be (batch_size, seq_len), with True values for positions to ignore
            padding_masks = ~masks.t()
        else:
            # Inference mode
            x = x.unsqueeze(0)  # Adjust for batch dimension in inference
            padding_masks = None
            print(f"Input size in inference mode: {x.size}")
        
        if seq_len > self.max_seq_len:
            # Generate a mask to limit attention to the last 'max_seq_len' tokens
            causal_mask = self.generate_sliding_window_causal_mask(seq_len, device=x.device)
            print(f"sliding window causal mask: {causal_mask}")
        else:
            # Standard causal mask since the sequence length is within the window
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            print(f"standard causal mask: {causal_mask}")

        # Embed the input (seq_len, batch_size, num_obs) --> (seq_len, batch_size, d_model)
        x = self.embedding(x)
        x = self.pos_encoder(x) # (seq_len, batch_size, d_model)

        # Pass through the transformer.
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_masks)   # (seq_len, batch_size, d_model)
        if padding_masks is not None:
            x = unpad_trajectories(x, padding_masks)
        print(f"Output size in inference mode: {x.size}")

        return x
    
    def generate_sliding_window_causal_mask(self, sz: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate a square causal mask to limit attention to the last max_seq_len tokens without loops.

        Args:
            sz (int): Size of the sequence (and the square mask).
            max_seq_len (int): Maximum number of tokens a token can attend to from its past.
            device (torch.device): The device on which to create the mask.

        Returns:
            torch.Tensor: The sliding window causal mask.
        """
        # Create a matrix where each element is its column index
        cols = torch.arange(sz, device=device).repeat(sz, 1)

        # Create a matrix where each element is its row index
        rows = torch.arange(sz, device=device).unsqueeze(1).repeat(1, sz)

        # Calculate the difference between each row and column. Mask should be zero if the difference is less than max_seq_len and more than or equal to 0
        sliding_window_mask = (rows - cols).float()

        # Apply the conditions for the sliding window
        mask = torch.where((sliding_window_mask >= 0) & (sliding_window_mask < self.max_seq_len), 0., float('-inf'))

        return mask
