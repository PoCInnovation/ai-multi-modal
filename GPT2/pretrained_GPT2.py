import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    layer_size : int = 768 # Model dimensions
    block_size : int = 256
    vocab_size : int = 50304
    number_block : int = 12
    number_head_attention : int = 12
    context_size : int = 1024



class CausalSelfAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.q = nn.Linear(config.layer_size, config.layer_size)
        self.k = nn.Linear(config.layer_size, config.layer_size)
        self.v = nn.Linear(config.layer_size, config.layer_size)
        self.softmax = nn.Softmax()
        # self.concatenate = nn.Linear(config.layer_size, config.layer_size * 3) # FIXME: Not valid 
        self.projection = nn.Linear(config.layer_size * 3, config.layer_size)

    def forward(self, x):
        self.q, self.k, self.v = None, None, None
        result = (self.q @ self.k).view(-1, 1)
        result_value = (result @ self.v).view(-1, 1)
        
        pass
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.concat_attention = nn.Linear(config.layer_size * 3, config.layer_size)
        self.softmax = nn.Softmax()
        self.size = config.layer_size
        self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size))
                             .view(1, 1, config.context_size, config.context_size))

        self.attn_head = config.number_head_attention
        self.head_size = self.size // self.attn_head # 64
        self.proj = nn.Linear(self.size, self.size)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.concat_attention(x)
        q, k, v = qkv.split(self.size, dim=2)

        # reshape de matrice
        k = k.view(B, T, self.attn_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.attn_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.attn_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # multiplication de q et k :
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        # le tril de la matrice 
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))   

        # softmax
        attn = self.softmax(attn, dim=-1)

        # multiplication de attn et v
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C) # reassamble attention heads
        out = self.proj(out)

        return out


        
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
        































