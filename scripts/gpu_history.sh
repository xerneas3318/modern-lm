#!/usr/bin/env bash
# gpu_history.sh — timestamped GPU sample every 60s, so saturation can be audited
# over the whole run rather than trusting instantaneous spot checks.
while true; do
  printf '%s,%s\n' "$(date -u '+%H:%M:%S')" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader | tr -d ' ')" \
    >> /workspace/gpu_history.log
  sleep 60
done
