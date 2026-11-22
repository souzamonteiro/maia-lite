"""MaiaFormer-355 v0.1: minimal Llama-compatible architecture in PyTorch.
Research prototype, not an optimized trainer. Requires torch.
"""
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 65536
    hidden_size: int = 896
    intermediate_size: int = 3072
    num_hidden_layers: int = 29
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def validate(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.max_position_embeddings > 0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight

def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)

def apply_rope(q, k, positions, theta):
    d = q.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    ang = torch.outer(positions.float(), inv)
    emb = torch.cat((ang, ang), dim=-1)
    cos = emb.cos()[None, None].to(q.dtype)
    sin = emb.sin()[None, None].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, nh, nk = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = h // nh
        self.nh, self.nk, self.theta = nh, nk, cfg.rope_theta
        self.q_proj = nn.Linear(h, nh*self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, nk*self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, nk*self.head_dim, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
    def forward(self, x):
        b, t, _ = x.shape
        shape = lambda y, n: y.view(b, t, n, self.head_dim).transpose(1, 2)
        q, k, v = shape(self.q_proj(x), self.nh), shape(self.k_proj(x), self.nk), shape(self.v_proj(x), self.nk)
        q, k = apply_rope(q, k, torch.arange(t, device=x.device), self.theta)
        repeat = self.nh // self.nk
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, f = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, f, bias=False)
        self.up_proj = nn.Linear(h, f, bias=False)
        self.down_proj = nn.Linear(f, h, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))

class MaiaFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))

def param_count(cfg):
    cfg.validate()
    h, d, nk, f, l, v = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size
    per_layer = (2*h*h + 2*h*nk*d) + 3*h*f + 2*h
    return v*h + l*per_layer + h + (0 if cfg.tie_word_embeddings else v*h)

def hf_config(cfg):
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            **asdict(cfg), "hidden_act": "silu", "attention_bias": False,
            "mlp_bias": False, "attention_dropout": 0.0, "initializer_range": 0.02,
            "torch_dtype": "float32", "bos_token_id": 1, "eos_token_id": 2,
            "pad_token_id": None, "use_cache": True}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tiny', action='store_true', help='Execute forward/backward on CPU with tiny architecture')
    p.add_argument('--write-config', type=Path)
    a = p.parse_args()
    cfg = Config() if not a.tiny else Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    print('Parameters:', f'{param_count(cfg):,}')
    if a.write_config:
        a.write_config.parent.mkdir(parents=True, exist_ok=True)
        a.write_config.write_text(json.dumps(hf_config(cfg), indent=2) + '\n')
    if a.tiny:
        torch.manual_seed(42)
        model = MaiaFormer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
        loss.backward()
        assert sum(p.numel() for p in model.parameters()) == param_count(cfg)
        assert torch.isfinite(loss)
        print('Smoke test OK:', logits.shape, 'loss=', round(loss.item(), 5))
