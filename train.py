"""
train.py — headless training for the modern GPT (RunPod / multi-GPU).

Same model + training logic as gpt.ipynb, packaged as a script:
  - no SMOKE branches (always the full run)
  - torch.compile always on
  - run-namespaced checkpointing under --out-dir/<run-name>/checkpoints
  - periodic generation samples + clean logging

Launch (dual GPU):
    torchrun --standalone --nproc_per_node=2 train.py --data-root /workspace/data

Single GPU:
    python train.py --data-root /workspace/data
"""

import os
import csv
import glob
import time
import math
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # data / io (RunPod-friendly defaults: persistent volume lives at /workspace)
    p.add_argument("--data-root", default="/workspace/data",
                   help="dir containing edu_fineweb10B/ shards")
    p.add_argument("--out-dir", default="/workspace/runs",
                   help="run artifacts (checkpoints, logs, samples) go under out-dir/<run-name>/")
    p.add_argument("--run-name", default="modern_1024x24")
    # data / batch
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64, help="per-GPU micro-batch B")
    p.add_argument("--total-batch-size", type=int, default=524288, help="tokens per optimizer step")
    # schedule
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=6e-4 * 2)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--decay-frac", type=float, default=0.20)
    # cadence
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--gen-every", type=int, default=4000)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--vocab-size", type=int, default=50304, help="padded vocab (real gpt2 is 50257)")
    return p.parse_args()


args = get_args()


# ----------------------------------------------------------------------------
# DDP
# ----------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP needs CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"

# run directory + logging
run_dir = os.path.join(args.out_dir, args.run_name)
ckpt_dir = os.path.join(run_dir, "checkpoints")
log_file = os.path.join(run_dir, "train.log")
if master_process:
    os.makedirs(ckpt_dir, exist_ok=True)


def log(msg, console=False):
    """Append to the run log; optionally echo to stdout. Master rank only."""
    if not master_process:
        return
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    if console:
        print(msg, flush=True)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")

enc = tiktoken.get_encoding("gpt2")

T = args.seq_len
B = args.batch_size
total_batch_size = args.total_batch_size
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must divide B*T*world"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)


# ----------------------------------------------------------------------------
# model  (verbatim from gpt.ipynb)
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

    def forward(self, x, ve):
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

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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

    def forward(self, x, ve):
        x = x + self.attn(self.ln_1(x), ve)
        x = x + self.ffw(self.ln_2(x))
        return x


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24
    n_embd: int = 1024
    n_head: int = 16
    kv_group: int = 4


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half = config.n_layer // 2
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
        skips = []
        x = self.transformer.wte(idx)
        ve = self.value_emb(idx)
        for i, block in enumerate(self.transformer.h):
            if i < self.half:
                x = block(x, ve)
                skips.append(x)
            else:
                x = x + self.skip_weights[i - self.half] * skips.pop()
                x = block(x, ve)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        cap = 30
        logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def configure_optimizers(model, muon_lr, adam_lr, weight_decay):
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
        lr=adam_lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device_type == "cuda"),
    )
    return muon, adamw


# ----------------------------------------------------------------------------
# data  (verbatim from gpt.ipynb)
# ----------------------------------------------------------------------------
SHARD_DIR = os.path.join(args.data_root, "edu_fineweb10B")


def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root=SHARD_DIR):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards for split {split} in {data_root}"
        if master_process:
            log(f"found {len(self.shards)} shards for split {split}", console=True)
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                              num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                            num_processes=ddp_world_size, split="val")


# ----------------------------------------------------------------------------
# build model, optimizers, schedulers
# ----------------------------------------------------------------------------
model = GPT(GPT2Config(vocab_size=args.vocab_size))
model.to(device)

max_steps = args.max_steps
stable_start = int(args.warmup_frac * max_steps)
decay_steps = int(args.decay_frac * max_steps)
decay_start = max_steps - decay_steps


def wsd(step):
    if step < stable_start:
        return (step + 1) / stable_start
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / decay_steps
    return 1.0 - prog


muon, adamw = configure_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                                   weight_decay=args.weight_decay)
optimizers = [muon, adamw]
schedulers = [torch.optim.lr_scheduler.LambdaLR(o, wsd) for o in optimizers]

model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
save_model = getattr(raw_model, "_orig_mod", raw_model)  # uncompiled, clean state_dict keys + fast gen

n_params = sum(p.numel() for p in save_model.parameters())
log(f"run={args.run_name} params={n_params/1e6:.1f}M world={ddp_world_size} "
    f"B={B} grad_accum={grad_accum_steps} total_batch={total_batch_size} "
    f"muon_lr={args.muon_lr} adam_lr={args.adam_lr} max_steps={max_steps}", console=True)


# resume from latest checkpoint in this run's dir (namespaced by run-name)
tr_loss = []
start_step = 0
_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
if _ckpts:
    _ckpt = torch.load(_ckpts[-1], map_location=device, weights_only=False)
    save_model.load_state_dict(_ckpt["model"])
    for o, sd in zip(optimizers, _ckpt["optimizer"]):
        o.load_state_dict(sd)
    for s, sd in zip(schedulers, _ckpt["scheduler"]):
        s.load_state_dict(sd)
    start_step = _ckpt["step"] + 1
    tr_loss = _ckpt.get("tr_loss", [])
    if "current_shard" in _ckpt:
        train_loader.current_shard = _ckpt["current_shard"]
        train_loader.current_position = _ckpt["current_position"]
        train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    log(f"resumed from {_ckpts[-1]} at step {start_step}", console=True)


@torch.no_grad()
def generate(prompt="Shakespeare:", n_new=60, k_samples=2):
    """Sample from the uncompiled model (master only) so torch.compile doesn't recompile per length."""
    save_model.eval()
    toks = enc.encode(prompt)
    outs = []
    for _ in range(k_samples):
        context = list(toks)
        for _ in range(n_new):
            logits, _ = save_model(torch.tensor([context[-T:]], device=device))
            logits = logits[:, -1, :]
            logits[:, 50257:] = float("-inf")   # mask padded vocab
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context.append(ix)
        outs.append(enc.decode(context[len(toks):]))
    save_model.train()
    return outs


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
model.train()
last_val_loss = float("nan")
t0 = time.time()

for step in range(start_step, max_steps):
    last_step = (step == max_steps - 1)
    tstep = time.time()

    # validation
    if step % args.val_every == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_accum += (loss / val_steps).detach()
        if ddp:
            dist.all_reduce(val_accum, op=dist.ReduceOp.AVG)
        last_val_loss = val_accum.item()
        log(f"{step} val {last_val_loss:.6f}")
        log(f"step {step:5d} | val loss {last_val_loss:.4f}", console=True)
        model.train()

    # optimizer step (grad accumulation)
    for o in optimizers:
        o.zero_grad()
    loss_accum = 0.0
    for micro in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in optimizers:
        o.step()
    for s in schedulers:
        s.step()

    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - tstep
    toks_per_sec = total_batch_size / dt if dt > 0 else 0.0

    if master_process:
        tr_loss.append(loss_accum.item())
        log(f"{step} train {loss_accum.item():.6f} norm {norm:.4f}")
        if step % 100 == 0 or last_step:
            lr_m = schedulers[0].get_last_lr()[0]
            lr_a = schedulers[1].get_last_lr()[0]
            log(f"step {step:5d} | loss {loss_accum.item():.4f} | lr_muon {lr_m:.2e} "
                f"lr_adam {lr_a:.2e} | norm {norm:.2f} | {toks_per_sec:,.0f} tok/s", console=True)

    # periodic generation sample
    if (step % args.gen_every == 0 and step > 0) or last_step:
        if master_process:
            log(f"--- samples @ step {step} ---", console=True)
            for s_i, sample in enumerate(generate()):
                log(f"[sample {s_i}] {sample!r}", console=True)
        if ddp:
            dist.barrier()

    # checkpoint (namespaced by run-name; keep the most recent few)
    if master_process and step > 0 and (step % args.ckpt_every == 0 or last_step):
        path = os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt")
        torch.save({
            "model": save_model.state_dict(),
            "optimizer": [o.state_dict() for o in optimizers],
            "scheduler": [s.state_dict() for s in schedulers],
            "step": step,
            "tr_loss": tr_loss,
            "current_shard": train_loader.current_shard,
            "current_position": train_loader.current_position,
            "config": asdict(save_model.config),
            "run_name": args.run_name,
        }, path)
        for old in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))[:-args.keep_ckpts]:
            os.remove(old)
        log(f"saved checkpoint -> {path}", console=True)


# final run identity
if master_process:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    steps_run = max_steps - start_step
    tokens_per_sec = steps_run * total_batch_size / elapsed if elapsed > 0 else 0.0
    log(f"run {args.run_name} | val loss {last_val_loss:.4f} | {tokens_per_sec:.0f} tok/s", console=True)
    results_csv = os.path.join(run_dir, "results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_name", "val_loss", "tokens_per_sec"])
        w.writerow([args.run_name, f"{last_val_loss:.6f}", f"{tokens_per_sec:.2f}"])

if ddp:
    destroy_process_group()
