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

        if masks is not None:
            # Training mode
            # Padding mask should be (batch_size, seq_len), with True values for positions to ignore (padding)
            padding_masks = ~masks.t()
            
            # Generate a causal mask to ensure the self-attention only attends to preceding tokens
            causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0), x.device)
            print(f"padding mask device: {padding_masks.device}")
        else:
            # inference mode
            x = x.unsqueeze(1)
            padding_masks = None
            
            # For inference, generate a mask for the maximal possible sequence length
            causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len, x.device)[:x.size(0), :x.size(0)]
            print(f"Inference causal mask shape: {causal_mask.shape}")
        print(f"causal mask devide: {causal_mask.device}")


        # Embed the input (seq_len, batch_size, num_obs)
        x = self.embedding(x)
        print(f"Embedded input shape: {x.shape}")
        x = self.pos_encoder(x) # (seq_len, batch_size, d_model)
        print(f"Embedded input with PE shape: {x.shape}")

        # Pass through the transformer. Note that we use a causal tranformer encoder here so that the self-attention
        # only attends to preceding tokens.
        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_masks, is_causal=True)   # (seq_len, batch_size, d_model)
        print(f"Transformer output shape: {out.shape}")
        if masks is not None:
            out = unpad_trajectories(out, masks)
            print(f"Transformer unpadded trajectory in training mode: {out.shape}")
        else:
            out = out.transpose(0, 1)
            print(f"Transformer output shape in inference mode: {out.shape}")   # torch.Size([4096, 1, 512])

        return out
