#!/usr/bin/env bash
#
# launch_train.sh — start (or RESUME) the modern_1024x24 pretraining run.
#
# Re-running this is the documented recovery path (train.md §3): train.py picks up
# the newest checkpoint in the run dir automatically. Safe to run repeatedly.
#
#   nohup setsid bash /workspace/launch_train.sh &
#
# Config deviations from train.md §1, both forced by this pod (see train.md APPENDIX):
#   --nproc_per_node=1   only 1x H100 80GB is visible, not 2x Pro 6000 96GB
#   --batch-size 16      B=64/32/16-uncompiled all OOM; 16+compile peaks at 62.7/79.2 GiB
# total_batch stays 524288 (grad_accum 16), so training dynamics are unchanged.
#
set -uo pipefail
source /workspace/.secrets/train-env.sh
cd /workspace/modern-lm

CONSOLE=/workspace/train_console.log
FW=/workspace/data/edu_fineweb10B
CODE=/workspace/data/code_python

# ---- preflight: refuse to start on incomplete data -------------------------
# Launching early is silently harmful: DataLoaderLite logs "(no train shards ...)"
# and renormalises the weights, so the run would train with NO code data at all
# instead of the intended 18%.
fw_train=$(ls "$FW"/edufineweb_train_*.npy 2>/dev/null | wc -l)
fw_val=$(ls "$FW"/edufineweb_val_*.npy 2>/dev/null | wc -l)
code_train=$(ls "$CODE"/code_train_*.npy 2>/dev/null | wc -l)

echo "preflight: fineweb train=$fw_train val=$fw_val | code train=$code_train" | tee -a "$CONSOLE"
if [ "$fw_train" -lt 99 ] || [ "$fw_val" -lt 1 ] || [ "$code_train" -lt 19 ]; then
  echo "ABORT: data incomplete (want fineweb>=99 train +1 val, code>=19 train)." | tee -a "$CONSOLE"
  exit 1
fi

echo "=== launch $(date -u '+%Y-%m-%d %H:%M:%SZ') ===" >> "$CONSOLE"
# MUST be the venv's torchrun. Bare `torchrun` resolves to /usr/local/bin/torchrun
# (system python3.12 + torch 2.8.0), which has no tqdm and is the wrong torch.
exec /workspace/modern-lm/.venv/bin/torchrun --standalone --nproc_per_node=1 train.py \
  --data-root /workspace/data \
  --code-dir "$CODE" --code-frac 0.18 \
  --run-name modern_1024x24 \
  --batch-size 16 \
  --log-every 10 >> "$CONSOLE" 2>&1
