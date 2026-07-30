"""
model.py -- shared model / optimizer / tokenizer definitions for modern-lm.

Imported by train.py (pretraining), sft.py (SFT + tool-call training), and
toolcall.py (tool-calling inference). Pure definitions: no argparse, no DDP,
no top-level side effects, so it's safe to import anywhere.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ----------------------------------------------------------------------------
# tokenizer + reserved special tokens
# ----------------------------------------------------------------------------
# Vocab is padded 50257 -> 50304, leaving IDs 50257..50303 free (rows the model
# already has embeddings for). We reserve a handful as atomic special tokens so
# the SFT / DPO / GRPO / tool-use stages need ZERO embedding surgery. They never
# appear in pretraining data (rows stay near-init, trained at SFT) -- expected.
# <|endoftext|> (50256) is the document separator (already in the FineWeb shards).
EOS_ID = 50256
SPECIAL_TOKENS = {
    "<|im_start|>": 50257,   # chat: turn start (ChatML)
    "<|im_end|>":   50258,   # chat: turn end
    "<pad>":        50259,   # padding for batched SFT
    "<think>":      50260,   # reasoning delimiter (GRPO / R1-style CoT)
    "</think>":     50261,
    "<tool_call>":  50262,   # tool / program-aided-math call
    "</tool_call>": 50263,
    "<tool_result>":  50264, # injected calculator result (model reads, never generates)
    "</tool_result>": 50265,
    # 50266..50303 remain free
}


def build_enc():
    """GPT-2 BPE + the reserved special tokens. Same merges/IDs as plain gpt2
    (so the pre-tokenized shards stay valid); just adds the atomic specials."""
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_math",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
    )


# ----------------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------------
def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)


# ----------------------------------------------------------------------------
# Muon
# ----------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B_ = b * A + c * (A @ A)
        X = a * X + B_ @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)
                g = zeropower_via_newtonschulz5(g, group["ns_steps"])
                if wd:
                    p.mul_(1 - lr * wd)
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g, alpha=-lr * scale)


# ----------------------------------------------------------------------------
# norms + attention + MLP + block + GPT
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class RMSNormHead(nn.Module):
    def __init__(self, config, eps=1e-6):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        self.weight = nn.Parameter(torch.ones(config.n_head, 1, head_dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.kv_group = config.kv_group
        assert config.n_embd % self.kv_group == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.kv_head = self.n_head // self.kv_group

        self.c_attn = nn.Linear(config.n_embd, config.n_embd + (2 * self.kv_head * self.head_dim), bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.ZERO_INIT = 1

        cos, sin = build_rope_cache(config.block_size, self.n_embd // self.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.q_norm = RMSNormHead(config)
        self.k_norm = RMSNorm(self.head_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.5]))

    def forward(self, x, ve, attn_mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_embd, self.kv_head * self.head_dim, self.kv_head * self.head_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])

        ve = ve.view(B, T, self.kv_head, self.head_dim).transpose(1, 2)
        v = self.lambdas[0] * v + self.lambdas[1] * ve

        k = k.repeat_interleave(self.kv_group, dim=1)
        v = v.repeat_interleave(self.kv_group, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class Swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dim = config.n_embd * 4
        self.w1 = nn.Linear(n_embd, dim, bias=False)
        self.w2 = nn.Linear(n_embd, dim, bias=False)
        self.w3 = nn.Linear(dim, n_embd, bias=False)
        self.w3.ZERO_INIT = 1

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffw = Swiglu(config)

    def forward(self, x, ve, attn_mask=None):
        x = x + self.attn(self.ln_1(x), ve, attn_mask)
        x = x + self.ffw(self.ln_2(x))
        return x


@dataclass
class GPT2Config:
    block_size: int = 2048
    vocab_size: int = 50304
    n_layer: int = 24
    n_embd: int = 1024
    n_head: int = 16
    kv_group: int = 4


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half = config.n_layer // 2
        self.doc_mask = False   # set True to enable intra-doc attention masking
        self.skip_weights = nn.Parameter(torch.ones(self.half))
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.value_emb = nn.Embedding(config.vocab_size, config.n_embd // config.kv_group)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "ZERO_INIT"):
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        attn_mask = None
        if self.doc_mask:
            doc_id = (idx == EOS_ID).cumsum(1)
            same_doc = doc_id[:, :, None] == doc_id[:, None, :]
            causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
            attn_mask = (same_doc & causal)[:, None, :, :]

        skips = []
        x = self.transformer.wte(idx)
        ve = self.value_emb(idx)
        for i, block in enumerate(self.transformer.h):
            if i < self.half:
                x = block(x, ve, attn_mask)
                skips.append(x)
            else:
                x = x + self.skip_weights[i - self.half] * skips.pop()
                x = block(x, ve, attn_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        cap = 15   # final-logit softcap (Gemma-style, modded-nanogpt tuned value)
        logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def configure_optimizers(model, muon_lr, adam_lr, weight_decay, fused=False):
    """Muon for hidden 2D matrices; AdamW for embeddings / lm_head / norms.
    Embeddings and lm_head are excluded from Muon by identity (the classic bug)."""
    wte = model.transformer.wte.weight
    lm_head = model.lm_head.weight
    value_emb = model.value_emb.weight

    muon_params, adam_decay, adam_nodecay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and p is not wte and p is not lm_head and p is not value_emb:
            muon_params.append(p)
        elif p is wte or p is lm_head or p is value_emb:
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)

    muon = Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(
        [{"params": adam_decay, "weight_decay": weight_decay},
         {"params": adam_nodecay, "weight_decay": 0.0}],
        lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, fused=fused,
    )
    return muon, adamw
