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
            num_actor_obs=d_model,
            num_critic_obs=d_model,
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

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.num_heads = num_heads
        # Make sure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.d_k = d_model // num_heads # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, padding_mask, dropout: nn.Dropout):
        batch_size, h, seq_len, d_k = query.shape
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_((mask==0).unsqueeze(1), -1e9)
        if padding_mask is not None:
            # padding mask: (seq_len, batch, 1) --> (batch, seq_len)
            padding_mask = padding_mask.transpose(0, 1).squeeze(-1)
            padding_mask = padding_mask.unsqueeze(1).expand(batch_size, h, seq_len)
            
            # Create a mask for the last element in the sequence
            last_elem_mask = torch.zeros_like(attention_scores).bool()
            last_elem_mask[:, :, -1, :] = (padding_mask==0)

            attention_scores.masked_fill_(last_elem_mask, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask, padding_mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, padding_mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[None, :, :]
    
class ExtendedEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, intermediate_dim=None, activation=nn.ELU()):
        super(ExtendedEmbedding, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = d_model  # Optionally set the intermediate dimension

        # First linear layer maps from input_dim to intermediate_dim
        self.linear1 = nn.Linear(input_dim, intermediate_dim)
        
        # Second linear layer maps from intermediate_dim to d_model
        self.linear2 = nn.Linear(intermediate_dim, intermediate_dim)

        self.linear3 = nn.Linear(intermediate_dim, d_model)

        # Activation function, e.g., ReLU
        self.activation = activation


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask, padding_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask, padding_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask, padding_mask):
        for layer in self.layers:
            x = layer(x, mask, padding_mask)
        return self.norm(x)

class TransformerMemory(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()

        # self.embedding = nn.Linear(input_dim, d_model)
        # d_model = input_dim
        self.embedding = ExtendedEmbedding(input_dim, d_model, intermediate_dim=512)
        self.pos_encoder = PositionalEmbedding(d_model)
        self.drop = torch.nn.Dropout(dropout)
        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(num_layers):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        
        self.transformer_encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        self.initialize_transformer()
    
    def initialize_transformer(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:  # Applies to weights of linear layers and not biases
                nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0
    
    def forward(self, x, masks=None, reset_masks=None):

        x = x.permute(1, 0, 2) # (seq_len, batch, num_obs) --> (batch, seq_len, num_obs)
        # x = x.unsqueeze(1)  # (batch, num_obs) --> (batch, seq_len, num_obs)
        seq_len = x.shape[1]

        # Generate a causal mask to limit attention to the preceding tokens
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)   # (1, seq_len, seq_len)

        # Embed the input (batch, seq_len, num_obs) --> (batch, seq_len, d_model)
        x = self.embedding(x)

        pos_ips = torch.arange(seq_len - 1, -1, -1.0, dtype=torch.float).to(
            x.device
        )
        
        x = x + self.pos_encoder(pos_ips)    # (batch x seq_len x d_model)

        # Pass through the transformer.
        x = self.transformer_encoder(x, mask=causal_mask, padding_mask=reset_masks) #, mask=causal_mask)   # (batch, seq_len, d_model)
        # x = x.squeeze(1)    # (batch, seq_len, d_model) --> (batch, d_model)
        x = x.permute(1, 0, 2) # (batch, seq_len, d_model) --> (seq_len, batch, d_model)
        
        if masks is not None:
            x = unpad_trajectories(x, masks)
        else:
            x = x[-1]   # take only the last output in inference mode

        return x
