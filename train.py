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
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import build_enc, GPT, GPT2Config, configure_optimizers
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
    p.add_argument("--seq-len", type=int, default=2048, help="context length (Tier 1: was 1024)")
    p.add_argument("--batch-size", type=int, default=32, help="per-GPU micro-batch B (dropped 64->32 for 2048 ctx)")
    p.add_argument("--total-batch-size", type=int, default=524288, help="tokens per optimizer step")
    p.add_argument("--doc-mask", action="store_true",
                   help="intra-document attention masking (block-diagonal on EOS). "
                        "OFF by default: enabling it disables the FlashAttention causal fast path, so it is slower.")
    # schedule
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=6e-4 * 2)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--decay-frac", type=float, default=0.20)
    # cadence
    p.add_argument("--log-every", type=int, default=100,
                   help="how often (in steps) to write the progress+ETA line to train.log/console. "
                        "Cheap: one line costs ~2.2ms on the network volume, so even 10 is ~4s "
                        "over the whole run.")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--gen-every", type=int, default=4000)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--vocab-size", type=int, default=50304, help="padded vocab (real gpt2 is 50257)")
    # model size (defaults = the 480M config; override for smoke tests on small GPUs)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--n-embd", type=int, default=1024)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--kv-group", type=int, default=4)
    # data mixing: mix a code corpus in with the web data at train time
    p.add_argument("--fineweb-dir", default=None, help="override the FineWeb shard dir (default <data-root>/edu_fineweb10B)")
    p.add_argument("--code-dir", default=None, help="dir of code shards to mix in (e.g. <data-root>/code_python)")
    p.add_argument("--code-frac", type=float, default=0.0, help="fraction of TRAIN batches drawn from --code-dir")
    # misc
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile (faster startup for smoke tests)")
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
        tqdm.write(msg)   # plays nicely with the live progress bar


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")

# tokenizer (GPT-2 BPE + reserved chat/reasoning/tool special tokens) lives in model.py
enc = build_enc()

T = args.seq_len
B = args.batch_size
total_batch_size = args.total_batch_size
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must divide B*T*world"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)


# model / optimizer definitions are in model.py


# ----------------------------------------------------------------------------
# data  (verbatim from gpt.ipynb)
# ----------------------------------------------------------------------------
FINEWEB_DIR = args.fineweb_dir or os.path.join(args.data_root, "edu_fineweb10B")


def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class _ShardStream:
    """Sequential reader over the shards of one dir/split (the original loader)."""
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards for split {split} in {data_root}"
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

    def state(self):
        return {"shard": self.current_shard, "pos": self.current_position}

    def load(self, st):
        self.current_shard = st["shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = st["pos"]


class DataLoaderLite:
    """Multi-source loader. `sources` is a list of (dir, weight); batches are
    drawn from the sources by a deterministic weighted round-robin (largest-
    remainder), so a source with weight 0.18 supplies ~18% of batches. A single
    source (weight 1.0) reproduces the original behavior exactly."""
    def __init__(self, B, T, process_rank, num_processes, split, sources):
        self.streams, self.weights, self.names = [], [], []
        for root, w in sources:
            try:
                self.streams.append(_ShardStream(B, T, process_rank, num_processes, split, root))
                self.weights.append(w)
                self.names.append(os.path.basename(root.rstrip("/")))
            except AssertionError:
                if master_process:
                    log(f"  (no {split} shards in {root}, skipping)", console=True)
        assert self.streams, f"no sources with {split} shards"
        tot = sum(self.weights)
        self.weights = [w / tot for w in self.weights]
        if master_process:
            mix = ", ".join(f"{n}={w:.2f}" for n, w in zip(self.names, self.weights))
            log(f"{split} sources: {mix}", console=True)
        self.reset()

    def reset(self):
        for s in self.streams:
            s.reset()
        self._acc = [0.0] * len(self.streams)

    def next_batch(self):
        self._acc = [a + w for a, w in zip(self._acc, self.weights)]
        j = max(range(len(self._acc)), key=self._acc.__getitem__)
        self._acc[j] -= 1.0
        return self.streams[j].next_batch()

    def state(self):
        return {"streams": [s.state() for s in self.streams], "acc": list(self._acc)}

    def load(self, st):
        for s, ss in zip(self.streams, st["streams"]):
            s.load(ss)
        self._acc = list(st["acc"])


# train mixes FineWeb + optional code; val is FineWeb only (comparable across runs)
_train_sources = [(FINEWEB_DIR, 1.0)]
if args.code_dir and args.code_frac > 0:
    _train_sources = [(FINEWEB_DIR, 1.0 - args.code_frac), (args.code_dir, args.code_frac)]

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                              num_processes=ddp_world_size, split="train", sources=_train_sources)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                            num_processes=ddp_world_size, split="val", sources=[(FINEWEB_DIR, 1.0)])


# ----------------------------------------------------------------------------
# build model, optimizers, schedulers
# ----------------------------------------------------------------------------
model = GPT(GPT2Config(vocab_size=args.vocab_size, block_size=args.seq_len,
                       n_layer=args.n_layer, n_embd=args.n_embd,
                       n_head=args.n_head, kv_group=args.kv_group))
model.doc_mask = args.doc_mask   # intra-document attention masking (off unless --doc-mask)
model.to(device)

max_steps = args.max_steps
stable_start = max(1, int(args.warmup_frac * max_steps))   # clamp >=1 so tiny test runs don't divide by zero
decay_steps = max(1, int(args.decay_frac * max_steps))
decay_start = max_steps - decay_steps


def wsd(step):
    if step < stable_start:
        return (step + 1) / stable_start
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / decay_steps
    return 1.0 - prog


muon, adamw = configure_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                                   weight_decay=args.weight_decay, fused=(device_type == "cuda"))
optimizers = [muon, adamw]
schedulers = [torch.optim.lr_scheduler.LambdaLR(o, wsd) for o in optimizers]

if not args.no_compile:
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
    if "loader" in _ckpt:
        train_loader.load(_ckpt["loader"])
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
# smoothed per-step time, used for the ETA. Deliberately NOT a simple
# elapsed/steps average: the first step carries the torch.compile cost (~3 min),
# which would otherwise be amortised into every remaining step and inflate the
# ETA by an order of magnitude early in the run.
ema_dt = None

pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps,
            disable=not master_process, dynamic_ncols=True, desc="train")
for step in pbar:
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
    if step > start_step:   # skip the compile-laden first step
        ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt

    if master_process:
        # kept every step: in-memory only (no I/O), and it is what gives loss.png its
        # full-resolution curve — it rides along in the checkpoint as "tr_loss".
        tr_loss.append(loss_accum.item())
        lr_m = schedulers[0].get_last_lr()[0]
        lr_a = schedulers[1].get_last_lr()[0]
        # live per-step view on the progress bar
        pbar.set_postfix(loss=f"{loss_accum.item():.3f}", lr_m=f"{lr_m:.1e}",
                         lr_a=f"{lr_a:.1e}", norm=f"{norm:.2f}", tok_s=f"{toks_per_sec:,.0f}")
        # log every --log-every steps (default 100). One combined line, still parseable
        # as "<step> train <loss> norm <norm> ..." so existing tooling keeps working.
        if step % args.log_every == 0 or last_step:
            el = time.time() - t0
            # no honest rate yet on the very first step (it carries the compile), so
            # say so rather than printing a garbage number
            if ema_dt:
                rem = (max_steps - 1 - step) * ema_dt
                eta = f"{int(rem // 3600)}h{int(rem % 3600 // 60):02d}m"
            else:
                eta = "(compiling)"
            wall = f"{int(el // 3600)}h{int(el % 3600 // 60):02d}m"
            log(f"{step} train {loss_accum.item():.6f} norm {norm:.4f} "
                f"lr_muon {lr_m:.2e} lr_adam {lr_a:.2e} tok_s {toks_per_sec:.0f} "
                f"| {step + 1}/{max_steps} ({(step + 1) / max_steps * 100:.1f}%) "
                f"| elapsed {wall} | eta {eta}", console=True)

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
            "loader": train_loader.state(),
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
