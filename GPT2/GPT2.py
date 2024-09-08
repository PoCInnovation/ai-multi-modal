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
        self.nb_head = config.nb_head # number of heads
        self.embed_size = config.embed_size # embedding size
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)
        # split the query, key, and value into the number of heads
        k = k.view(B, T, self.nb_head, C // self.nb_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.nb_head, C // self.nb_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.nb_head, C // self.nb_head).transpose(1, 2) # (B, nh, T, hs)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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
# @dataclass
# class GPTConfig:
#     block_size: int = 1024 # max sequence length
#     vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
#     n_layer: int = 12 # number of layers
#     nb_head: int = 12 # number of heads
#     embed_size: int = 768 # embedding dimension

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

        # wte and lm_head need to be the same weight matrix (they share weights and we save 30% of model params)   
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            

    def forward(self, idx, targets):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
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

        # we are going to compute the loss only if there are targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, nb_head and embed_size are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, nb_head=12, embed_size=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, nb_head=16, embed_size=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, nb_head=20, embed_size=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, nb_head=25, embed_size=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('GPT2/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model run on {device}")
model = GPT(GPTConfig())
model = model.to(device)
model = torch.compile(model)

data_loader = DataLoaderLite(B=4, T=1024)
torch.set_float32_matmul_precision('high')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
import time
start = time.time()

for i in range(500):
    t0 = time.time()
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0)
    dt = (t1 - t0) * 1000
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.2f}")
end = time.time()

print(f"Final time : {end - start}")

