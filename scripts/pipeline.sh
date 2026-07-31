#!/usr/bin/env bash
#
# pipeline.sh — unattended driver: supervise pretraining -> SFT -> tool-call SFT.
#
# Phase 1 supervises the already-running pretraining job and RELAUNCHES it if the
# process dies (train.py auto-resumes from the newest checkpoint, verified working),
# so a transient crash costs at most ~1000 steps instead of the whole run.
# Phases 2 and 3 fire only after pretraining writes results.csv — its definitive
# completion marker (train.md §5) — so they can never start on a half-trained base.
#
#   nohup setsid bash /workspace/pipeline.sh &
#
set -uo pipefail
source /workspace/.secrets/train-env.sh
cd /workspace/modern-lm
PY=/workspace/modern-lm/.venv/bin/python

RUN=/workspace/runs/modern_1024x24
LOG=/workspace/pipeline.log
MAX_RESTARTS=10

say() { printf '%s | %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$*" >> "$LOG"; }

say "=========== pipeline start ==========="

# ---------------------------------------------------------------- phase 1
say "PHASE 1: supervising pretraining until $RUN/results.csv appears"
restarts=0
while [ ! -f "$RUN/results.csv" ]; do
  if ! pgrep -f "[t]rain.py --data-root" >/dev/null 2>&1; then
    # process gone and no results.csv => it died
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      say "FATAL: pretraining died and hit MAX_RESTARTS=$MAX_RESTARTS. Stopping."
      exit 1
    fi
    restarts=$((restarts+1))
    last=$(ls -1 "$RUN"/checkpoints/ckpt_*.pt 2>/dev/null | tail -1)
    say "pretraining not running (restart $restarts/$MAX_RESTARTS); newest ckpt: ${last:-none}"
    setsid nohup bash /workspace/launch_train.sh > /dev/null 2>&1 < /dev/null &
    sleep 180   # let compile finish before re-checking
  fi
  sleep 60
done
say "PHASE 1 done: $(tail -1 "$RUN/results.csv")"

BASE=$(ls -1 "$RUN"/checkpoints/ckpt_*.pt 2>/dev/null | tail -1)
if [ -z "$BASE" ]; then say "FATAL: no base checkpoint found"; exit 1; fi
say "base checkpoint: $BASE"

# ---------------------------------------------------------------- phase 2: SFT
# Chat SFT with the runbook's 15% synthetic tool-call mix (train.md §6).
if [ -f /workspace/runs/sft1/results.csv ]; then
  say "PHASE 2 already complete, skipping"
else
  say "PHASE 2: SFT -> runs/sft1"
  "$PY" -u sft.py \
    --base-ckpt "$BASE" \
    --out-dir /workspace/runs --run-name sft1 \
    --data HuggingFaceTB/smol-smoltalk --toolcall-frac 0.15 \
    --epochs 2 --lr 1e-4 --batch-size 8 >> "$LOG" 2>&1
  say "PHASE 2 exit=$?"
fi

SFT1=$(ls -1 /workspace/runs/sft1/checkpoints/sft_*.pt 2>/dev/null | tail -1)
if [ -z "$SFT1" ]; then say "FATAL: no sft1 checkpoint"; exit 1; fi
say "sft1 checkpoint: $SFT1"

# ---------------------------------------------------------------- phase 3: tool-call specialist
# Continues FROM the chat-tuned model with a heavier tool-call mix and a lower LR,
# so it specialises on tool use without unlearning chat. Deliberately NOT
# --toolcall-frac 1.0: that path skips the chat dataset entirely (sft.py:104) and
# would train on synthetic arithmetic only.
if [ -f /workspace/runs/sft2_toolcall/results.csv ]; then
  say "PHASE 3 already complete, skipping"
else
  say "PHASE 3: tool-call specialisation -> runs/sft2_toolcall"
  "$PY" -u sft.py \
    --base-ckpt "$SFT1" \
    --out-dir /workspace/runs --run-name sft2_toolcall \
    --data HuggingFaceTB/smol-smoltalk --toolcall-frac 0.50 \
    --max-steps 4000 --lr 5e-5 --batch-size 8 >> "$LOG" 2>&1
  say "PHASE 3 exit=$?"
fi

# ---------------------------------------------------------------- phase 4: QA
# Closed-book question answering. Branches from the CHAT model (not the tool-call
# one) so each specialist is an independent head on the same instruction-tuned base.
if [ -f /workspace/runs/sft3_qa/results.csv ]; then
  say "PHASE 4 already complete, skipping"
else
  say "PHASE 4: QA specialisation -> runs/sft3_qa"
  "$PY" -u sft.py \
    --base-ckpt "$SFT1" \
    --out-dir /workspace/runs --run-name sft3_qa \
    --data /workspace/sft_data/qa.jsonl --toolcall-frac 0.0 \
    --max-steps 3000 --lr 5e-5 --batch-size 8 --seq-len 1024 >> "$LOG" 2>&1
  say "PHASE 4 exit=$?"
fi

# ---------------------------------------------------------------- phase 5: RAG
# Grounded answering over supplied context, ~34% of examples unanswerable so the
# model learns to abstain rather than invent an answer.
if [ -f /workspace/runs/sft4_rag/results.csv ]; then
  say "PHASE 5 already complete, skipping"
else
  say "PHASE 5: RAG specialisation -> runs/sft4_rag"
  "$PY" -u sft.py \
    --base-ckpt "$SFT1" \
    --out-dir /workspace/runs --run-name sft4_rag \
    --data /workspace/sft_data/rag.jsonl --toolcall-frac 0.0 \
    --max-steps 4000 --lr 5e-5 --batch-size 8 --seq-len 1024 >> "$LOG" 2>&1
  say "PHASE 5 exit=$?"
fi

say "=========== pipeline COMPLETE ==========="
say "base:      $BASE"
say "sft1:      $(ls -1 /workspace/runs/sft1/checkpoints/sft_*.pt 2>/dev/null | tail -1)"
say "toolcall:  $(ls -1 /workspace/runs/sft2_toolcall/checkpoints/sft_*.pt 2>/dev/null | tail -1)"
say "qa:        $(ls -1 /workspace/runs/sft3_qa/checkpoints/sft_*.pt 2>/dev/null | tail -1)"
say "rag:       $(ls -1 /workspace/runs/sft4_rag/checkpoints/sft_*.pt 2>/dev/null | tail -1)"
