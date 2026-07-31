#!/usr/bin/env bash
#
# gpu_watchdog.sh — alert when the GPU is idle/underused while work is supposed to
# be running. Exits (which raises a notification) on the FIRST real anomaly.
#
# Naive "util < X" alarms are useless here: the GPU legitimately sits idle between
# pipeline stages, during HF uploads, and while a checkpoint is being written. So the
# watchdog only fires on states that are actually wrong:
#
#   A. a trainer process is ALIVE but the GPU has been under $MIN_UTIL% for
#      $STALL_SAMPLES consecutive samples  -> hung / stalled / fell off the GPU
#   B. NO trainer is running, the pipeline has not logged COMPLETE, and that has
#      persisted for $ORPHAN_SAMPLES samples -> a stage died without a relaunch
#
# Checkpoint writes stall the GPU for ~1 sample, so STALL_SAMPLES must stay well
# above 1 or every checkpoint would trip it.
#
#   bash /workspace/gpu_watchdog.sh          # runs until an anomaly, then exits
#
set -uo pipefail
MIN_UTIL=${MIN_UTIL:-40}
STALL_SAMPLES=${STALL_SAMPLES:-6}     # 6 x 30s = 3 min under-utilised
ORPHAN_SAMPLES=${ORPHAN_SAMPLES:-10}  # 10 x 30s = 5 min with nothing running
INTERVAL=${INTERVAL:-30}
LOG=/workspace/watchdog.log

say(){ printf '%s | %s\n' "$(date -u '+%H:%M:%SZ')" "$*" | tee -a $LOG; }
trainer_alive(){ pgrep -f "[t]rain\.py --data-root|[s]ft\.py" >/dev/null 2>&1; }
pipeline_done(){ grep -q "pipeline COMPLETE" /workspace/pipeline.log 2>/dev/null; }

say "watchdog armed (min_util=${MIN_UTIL}%, stall=${STALL_SAMPLES}, orphan=${ORPHAN_SAMPLES}, every ${INTERVAL}s)"
stall=0; orphan=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=${util:-0}
  if trainer_alive; then
    orphan=0
    if [ "$util" -lt "$MIN_UTIL" ]; then
      stall=$((stall+1))
      [ "$stall" -ge 3 ] && say "note: util ${util}% with trainer alive (${stall}/${STALL_SAMPLES})"
    else
      stall=0
    fi
    if [ "$stall" -ge "$STALL_SAMPLES" ]; then
      say "!!! ANOMALY A: trainer alive but GPU under ${MIN_UTIL}% for $((stall*INTERVAL))s"
      say "    procs: $(pgrep -af '[t]rain\.py|[s]ft\.py' | head -2)"
      say "    gpu  : $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
      exit 10
    fi
  else
    stall=0
    if pipeline_done; then
      say "pipeline COMPLETE and no trainer running — nothing left to watch, exiting cleanly"
      exit 0
    fi
    orphan=$((orphan+1))
    if [ "$orphan" -ge "$ORPHAN_SAMPLES" ]; then
      say "!!! ANOMALY B: no trainer running for $((orphan*INTERVAL))s and pipeline not COMPLETE"
      say "    last pipeline line: $(tail -1 /workspace/pipeline.log)"
      exit 11
    fi
  fi
  sleep "$INTERVAL"
done
