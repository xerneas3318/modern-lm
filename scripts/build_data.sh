#!/usr/bin/env bash
# build_data.sh — build both training corpora, sequentially.
#
# Sequential on purpose: both stages are CPU-bound on the same ~23.8-core cgroup
# quota, so running them together just halves each one's throughput.
#
# NOTE --nproc 22: prepare_data.py defaults to os.cpu_count()-2, which reports the
# HOST's 224 cores, not this container's 23.8-core quota. The default oversubscribes
# by ~10x and the build crawls.
#
#   nohup setsid bash /workspace/build_data.sh &
#
set -uo pipefail
source /workspace/.secrets/train-env.sh
cd /workspace/modern-lm
PY=/workspace/modern-lm/.venv/bin/python
NPROC=22
LOG=/workspace/data_build.log

say() { printf '\n===== %s | %s =====\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$*" >> "$LOG"; }

say "START build_data.sh (nproc=$NPROC)"

# ---- 1. FineWeb-Edu, 10B tokens (the general corpus) -----------------------
if ls /workspace/data/edu_fineweb10B/edufineweb_train_*.npy >/dev/null 2>&1 \
   && [ "$(ls /workspace/data/edu_fineweb10B/edufineweb_train_*.npy 2>/dev/null | wc -l)" -ge 99 ]; then
  say "fineweb already complete, skipping"
else
  say "fineweb-edu 10B -> /workspace/data/edu_fineweb10B"
  "$PY" -u prepare_data.py \
    --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --text-col text \
    --out-dir /workspace/data/edu_fineweb10B --prefix edufineweb \
    --total-tokens 10_000_000_000 --nproc "$NPROC" >> "$LOG" 2>&1
  say "fineweb exit=$?"
fi

# ---- 2. Python code, 2B tokens ---------------------------------------------
if ls /workspace/data/code_python/code_train_*.npy >/dev/null 2>&1 \
   && [ "$(ls /workspace/data/code_python/code_train_*.npy 2>/dev/null | wc -l)" -ge 19 ]; then
  say "code already complete, skipping"
else
  say "codeparrot-clean 2B -> /workspace/data/code_python"
  "$PY" -u prepare_data.py \
    --dataset codeparrot/codeparrot-clean --text-col content \
    --out-dir /workspace/data/code_python --prefix code \
    --total-tokens 2_000_000_000 --nproc "$NPROC" >> "$LOG" 2>&1
  say "code exit=$?"
fi

say "DONE. shard counts:"
{
  echo "  fineweb train: $(ls /workspace/data/edu_fineweb10B/edufineweb_train_*.npy 2>/dev/null | wc -l)"
  echo "  fineweb val:   $(ls /workspace/data/edu_fineweb10B/edufineweb_val_*.npy 2>/dev/null | wc -l)"
  echo "  code train:    $(ls /workspace/data/code_python/code_train_*.npy 2>/dev/null | wc -l)"
  du -sh /workspace/data 2>/dev/null
} >> "$LOG" 2>&1
