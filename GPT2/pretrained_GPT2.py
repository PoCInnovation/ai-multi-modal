import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class GPTConfig:
    layer_size : int = 768 # Model dimensions
    block_size : int = 256
    vocab_size : int = 50304
    number_block : int  = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.q = nn.Linear(config.layer_size, config.layer_size)
        self.k = nn.Linear(config.layer_size, config.layer_size)
        self.v = nn.Linear(config.layer_size, config.layer_size)
        self.softmax = nn.Softmax()
        self.concatenate = nn.Linear(config.layer_size, config.layer_size * 3)
        self.projection = nn.Linear(config.layer_size * 3, config.layer_size)

    def forward(self, x):
        self.q, self.k, self.v = None, None, None
        result = (self.q @ self.k).view(-1, 1)
        result_value = (result @ self.v).view(-1, 1)
        
        pass
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(config.layer_size, config.layer_size * 4)
        self.gelu = nn.GELU(approximate='tanh') # FIXME: REMOVE the aproximation
        self.linear2 = nn.Linear(config.layer_size * 4, config.layer_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class DecoderBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.layer_size)
        self.layer_norm_2 = nn.LayerNorm(config.layer_size)
        self.mlp = MLP()
        self.attention = CausalSelfAttention()

    def forward(self, x):
        x += self.attention(self.layer_norm_1(x))
        x += self.mlp(self.layer_norm_2(x))
        return x


class Model(nn.Model):
    def __init__(self):
        super().__init__(None)

        self.tokenizer = None
    
        self.decoder_block = DecoderBlock()
        































