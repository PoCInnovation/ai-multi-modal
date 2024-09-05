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
    vocab_size : int = 50257 # size of the vocabulary (50257 Byte Pair Encoding tokens + round up to the nearest multiple of 64)
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
        # TODO : replace attention by flash attention
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
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            wpe = nn.Embedding(config.block_size, config.embed_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embed_size),
        ))

        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
    
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # TODO: add a cross entropy loss (understand why loss is here)
        return logits
    
# -----------------------------------------------------------------
import tiktoken
import numpy as np
import os

# TODO: def load_tokens refers to whats ?
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_process, split):
        self.B = B
        self.T =T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        # TODO: implement the following line
        # if master_process:
        #     print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
