#!/usr/bin/env bash
#
# preserve_ckpts.sh — rescue milestone checkpoints from --keep-ckpts rotation.
#
# train.py keeps only the newest 3 checkpoints. The WSD end-of-stable checkpoint
# (step 15000 — the last one before LR decay begins at step 15259) would be DELETED
# when ckpt_018000.pt is written. This copies milestones to /workspace/keep/ as soon
# as they appear, well inside that window.
#
#   nohup setsid bash /workspace/preserve_ckpts.sh &
#
set -uo pipefail
CK=/workspace/runs/modern_1024x24/checkpoints
KEEP=/workspace/keep
LOG=/workspace/preserve.log
mkdir -p "$KEEP"

# step:label — end of WSD stable phase, and end of decay (final)
declare -A WANT=( [015000]=end_of_stable [019072]=end_of_decay )

say() { printf '%s | %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$*" >> "$LOG"; }
say "preserve daemon started; watching $CK"

remaining=${#WANT[@]}
while [ "$remaining" -gt 0 ]; do
  for step in "${!WANT[@]}"; do
    label="${WANT[$step]}"
    src="$CK/ckpt_${step}.pt"
    dst="$KEEP/ckpt_${step}_${label}.pt"
    if [ -f "$dst" ]; then continue; fi
    if [ -f "$src" ]; then
      # wait for the write to settle (4.3GB), then copy and verify size
      sz1=$(stat -c %s "$src" 2>/dev/null || echo 0); sleep 20
      sz2=$(stat -c %s "$src" 2>/dev/null || echo 0)
      if [ "$sz1" != "$sz2" ] || [ "$sz2" = 0 ]; then
        say "ckpt_${step}: still being written, retrying"; continue
      fi
      if cp "$src" "$dst.part" && mv "$dst.part" "$dst"; then
        say "PRESERVED ckpt_${step}.pt -> $(basename "$dst") ($(du -h "$dst" | cut -f1))"
        remaining=$((remaining-1))
      else
        say "ERROR copying ckpt_${step}.pt"; rm -f "$dst.part"
      fi
    fi
  done
  [ "$remaining" -gt 0 ] && sleep 60
done
say "all milestones preserved: $(ls "$KEEP")"
