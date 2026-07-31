"""
sft.py -- supervised fine-tuning (+ tool-calling) for the modern-lm model.

Turns the pretrained base into a chat/tool-using assistant:
  - loads the pretrained checkpoint (--base-ckpt),
  - trains on chat data (default HuggingFaceTB/smol-smoltalk) formatted as ChatML,
    with the loss MASKED to assistant tokens only (prompt + injected tool results
    are masked), via toolcall.format_chat / format_toolcall_example,
  - mixes in synthetic arithmetic tool-call examples (--toolcall-frac) so the model
    learns to emit <tool_call>expr</tool_call>,
  - saves an SFT checkpoint and runs a tool-calling demo.

Single GPU is plenty (SFT is cheap). Examples:
  # real SFT from a pretrained checkpoint
  python sft.py --base-ckpt /workspace/runs/modern_1024x24/checkpoints/ckpt_019072.pt \
      --out-dir /workspace/runs --run-name sft1 --epochs 2 --lr 1e-4

  # quick smoke (tiny fresh model, synthetic tool-call data only, no download)
  python sft.py --n-layer 2 --n-embd 128 --n-head 2 --kv-group 2 --seq-len 256 \
      --toolcall-frac 1.0 --max-steps 20 --batch-size 4 --no-compile
"""

import os
import csv
import glob
import time
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT, GPT2Config, build_enc, SPECIAL_TOKENS
from toolcall import format_chat, format_toolcall_example, generate_with_tools


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", default=None, help="pretrained checkpoint to start from (else fresh init)")
    p.add_argument("--out-dir", default="/workspace/runs")
    p.add_argument("--run-name", default="sft")
    p.add_argument("--data", default="HuggingFaceTB/smol-smoltalk", help="HF chat dataset")
    p.add_argument("--messages-col", default="messages")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps (0 = use epochs/data)")
    p.add_argument("--toolcall-frac", type=float, default=0.15, help="fraction of examples that are synthetic tool-call")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    # model size (only used if --base-ckpt not given, e.g. smoke tests)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--n-embd", type=int, default=1024)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--kv-group", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


args = get_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
random.seed(args.seed)
enc = build_enc()
PAD = SPECIAL_TOKENS["<pad>"]

run_dir = os.path.join(args.out_dir, args.run_name)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)


# ----------------------------------------------------------------------------
# model: load pretrained base (or fresh, for smoke tests)
# ----------------------------------------------------------------------------
if args.base_ckpt:
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    cfg = GPT2Config(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"loaded base from {args.base_ckpt}", flush=True)
else:
    cfg = GPT2Config(vocab_size=args.vocab_size, block_size=args.seq_len,
                     n_layer=args.n_layer, n_embd=args.n_embd,
                     n_head=args.n_head, kv_group=args.kv_group)
    model = GPT(cfg)
    print("no --base-ckpt: fresh init (smoke/test mode)", flush=True)
model.to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  device={device}", flush=True)


# ----------------------------------------------------------------------------
# data: chat examples (masked to assistant) + synthetic tool-call examples
# ----------------------------------------------------------------------------
SYS = "You are a helpful assistant. Use a calculator tool for arithmetic."
_OPS = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]


def synth_toolcall_example():
    a, b = random.randint(2, 999), random.randint(2, 99)
    sym, fn = random.choice(_OPS)
    ans = fn(a, b)
    q = f"What is {a} {sym} {b}?"
    return format_toolcall_example(enc, q, f"I'll compute {a} {sym} {b}.",
                                   f"{a}{sym}{b}", f"The answer is {ans}.", system=SYS)


def _mk_iter():
    """Fresh iterator over --data. Accepts a HF repo id OR a local .jsonl/.json file
    of {"messages": [...]} rows (used for the QA and RAG sets)."""
    from datasets import load_dataset
    if args.data.endswith((".jsonl", ".json")):
        return iter(load_dataset("json", data_files=args.data, split="train", streaming=True))
    return iter(load_dataset(args.data, split="train", streaming=True))


def chat_stream():
    """Yield (ids, mask) examples: mostly dataset chat, some synthetic tool-call."""
    ds_iter = None
    if args.toolcall_frac < 1.0:
        ds_iter = _mk_iter()
    while True:
        if ds_iter is None or random.random() < args.toolcall_frac:
            yield synth_toolcall_example()
        else:
            try:
                row = next(ds_iter)
            except StopIteration:
                # a local file is finite: start another epoch. Letting StopIteration
                # escape a generator raises RuntimeError under PEP 479, which would
                # kill the run at the end of epoch 1 instead of looping.
                ds_iter = _mk_iter()
                row = next(ds_iter)
            msgs = row[args.messages_col]
            ids, mask = format_chat(enc, msgs)
            if sum(mask) > 0:                      # skip if nothing to learn
                yield ids, mask


def batches(stream, B, T, steps):
    buf = []
    made = 0
    for ids, mask in stream:
        ids, mask = ids[:T], mask[:T]
        if len(ids) < 2:
            continue
        buf.append((ids, mask))
        if len(buf) == B:
            L = max(len(i) for i, _ in buf)
            X = torch.full((B, L), PAD, dtype=torch.long)
            M = torch.zeros((B, L), dtype=torch.long)
            for r, (i, m) in enumerate(buf):
                X[r, :len(i)] = torch.tensor(i); M[r, :len(m)] = torch.tensor(m)
            yield X, M
            buf = []; made += 1
            if steps and made >= steps:
                return


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                        weight_decay=args.weight_decay, fused=(device == "cuda"))
if not args.no_compile:
    model = torch.compile(model)

# rough step budget: max_steps if set, else a default cap (SFT is cheap; user tunes)
total_steps = args.max_steps if args.max_steps else 5000 * args.epochs


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    return args.lr  # flat after warmup (SFT is short)


model.train()
t0 = time.time()
pbar = tqdm(total=total_steps, desc="sft")
step = 0
for X, M in batches(chat_stream(), args.batch_size, args.seq_len, total_steps):
    X, M = X.to(device), M.to(device)
    x, y, m = X[:, :-1], X[:, 1:], M[:, 1:]
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"), dtype=torch.bfloat16):
        logits, _ = model(x)
    # masked next-token CE: loss only on assistant tokens
    loss_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               y.reshape(-1), reduction="none").view_as(y)
    denom = m.sum().clamp(min=1)
    loss = (loss_tok * m).sum() / denom
    opt.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    step += 1
    pbar.update(1)
    pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_at(step):.1e}", norm=f"{norm:.2f}")
    with open(os.path.join(run_dir, "sft.log"), "a") as f:
        f.write(f"{step} loss {loss.item():.6f} norm {norm:.4f}\n")

    if step % args.ckpt_every == 0 or step >= total_steps:
        save = getattr(model, "_orig_mod", model)
        path = os.path.join(run_dir, "checkpoints", f"sft_{step:06d}.pt")
        torch.save({"model": save.state_dict(),
                    "config": {"block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
                               "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                               "n_head": cfg.n_head, "kv_group": cfg.kv_group},
                    "step": step}, path)
        for old in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "sft_*.pt")))[:-3]:
            os.remove(old)
        tqdm.write(f"saved {path}")

pbar.close()
print(f"sft done: {step} steps, {time.time()-t0:.0f}s", flush=True)

# tool-calling demo
gen_model = getattr(model, "_orig_mod", model)
gen_model.eval()
for q in ["What is 47 * 89?", "A crate has 23 boxes of 17 apples. How many apples?"]:
    out = generate_with_tools(gen_model, enc, q, device, system=SYS, max_new=96,
                              block_size=cfg.block_size)
    print(f"\nQ: {q}\nA: {out!r}", flush=True)

with open(os.path.join(run_dir, "results.csv"), "a", newline="") as f:
    csv.writer(f).writerow([args.run_name, step, f"{time.time()-t0:.1f}"])
