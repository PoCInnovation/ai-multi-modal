import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import numpy as np
import math


"""
GPTConfig class

This class is used to store the configuration of the GPT model and is used to initialize the model.
"""
@dataclass
class GPTConfig:
    block_size : int = 1024 # max sequence length
    vocab_size : int = 50304 # size of the vocabulary (50257 Byte Pair Encoding tokens + round up to the nearest multiple of 64)
    embed_size : int = 768 # dimension of the embeddings
    n_layer : int = 12 # number of layers
    nb_head : int = 12 # number of heads in the multi-head attention

"""
CausalSelfAttention class

In this class we match the layers names of the original GPT2 model.
To be able to test the model by reusing the pretrained weights
"""
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.nb_head == 0 # embedding size must be divisible by the number of heads
        self.c_attn = nn.Linear(config.embed_size, config.embed_size * 3) # linear layer for the query, key, and value
        self.c_proj = nn.Linear(config.embed_size, config.embed_size) # output projection
        self.head_size = config.embed_size // config.nb_head # 768 // 12 = 64 (head size)
        # bias is actually a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # register_buffer is used to store a tensor that is not a model parameter, which means it won't be updated during training and won't be optimized by the optimizer (bias acts as a constant)
        self.n_head = config.nb_head # number of heads
        self.n_embd = config.embed_size # embedding size
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # split the query, key, and value into the number of heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
    
"""
MLP class

This class is used to define the feedforward neural network used in the GPT model.
"""
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, config.embed_size * 4)
        self.gelu = nn.GELU(approximate='tanh') #FIXME: remove the approximation
        self.c_proj = nn.Linear(config.embed_size * 4, config.embed_size)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
"""
Block class

This class is used to define the block of the GPT model.
"""
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLP(config)
        self.attn = CausalSelfAttention(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
"""
GPT class

This class is used to define the GPT model.
"""
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass
    