"""
make_loss_png.py — build the loss curve from a checkpoint's tr_loss.

Deliberately NOT parsing train.log the way train.md §5b does: logging is now every
10 steps, so the log holds ~1900 points while the checkpoint's "tr_loss" holds the
full per-step list (and survives resumes, since it rides along in the checkpoint).

  python make_loss_png.py <ckpt.pt> <out.png> [--title "..."]
"""
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--title", default="modern_1024x24 — pretraining loss")
ap.add_argument("--decay-start", type=int, default=15259)
a = ap.parse_args()

c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
tr = c["tr_loss"]
print(f"{len(tr)} points from {a.ckpt} (step {c['step']})")

plt.figure(figsize=(11, 5.5))
plt.plot(tr, lw=0.5, color="steelblue", alpha=0.65, label="train loss (per step)")
if len(tr) >= 100:
    k = 100
    s = np.convolve(tr, np.ones(k) / k, "valid")
    plt.plot(range(k - 1, len(tr)), s, color="crimson", lw=2, label=f"{k}-step moving avg")

# mark the WSD phase boundary — the reason the end-of-stable checkpoint is a
# distinct model from the final one
if a.decay_start < len(tr):
    plt.axvline(a.decay_start, color="darkgreen", ls="--", lw=1.4,
                label=f"decay starts (step {a.decay_start})")

plt.xlabel("step")
plt.ylabel("train loss")
plt.title(a.title)
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(bottom=min(tr) * 0.95)
plt.savefig(a.out, dpi=150, bbox_inches="tight")
print(f"wrote {a.out}")
