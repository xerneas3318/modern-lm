#!/usr/bin/env bash
#
# healthcheck.sh [WAIT_MIN] — wait up to WAIT_MIN minutes (exiting EARLY on trouble),
# then print a full health report on the training pipeline.
#
# Trouble = NaN/Inf loss, or the training process gone while pretraining is
# still unfinished. Exiting early matters: a silent death at hour 12 should not
# wait for the next scheduled check-in.
#
set -uo pipefail
WAIT_MIN="${1:-45}"
RUN=/workspace/runs/modern_1024x24
LOG=$RUN/train.log
TROUBLE=""

deadline=$(( $(date +%s) + WAIT_MIN*60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if grep -qiE " (nan|inf)$| (nan|inf) " "$LOG" 2>/dev/null; then TROUBLE="NaN/Inf in loss"; break; fi
  if [ ! -f "$RUN/results.csv" ] && ! pgrep -f "[t]rain.py --data-root" >/dev/null 2>&1; then
    # give the supervisor a moment to relaunch before crying wolf
    sleep 90
    if [ ! -f "$RUN/results.csv" ] && ! pgrep -f "[t]rain.py --data-root" >/dev/null 2>&1; then
      TROUBLE="training process gone (pretraining unfinished)"; break
    fi
  fi
  if [ -f "$RUN/results.csv" ]; then TROUBLE="PRETRAINING COMPLETE"; break; fi
  sleep 30
done

echo "================ HEALTH REPORT $(date -u '+%Y-%m-%d %H:%M:%SZ') ================"
[ -n "$TROUBLE" ] && echo "!!! EARLY EXIT: $TROUBLE" || echo "status: nominal (waited ${WAIT_MIN}m)"

# step comes from the live tqdm bar (authoritative, and independent of how often
# train.py writes to the log — logging is every 100 steps now).
step=$(tr '\r' '\n' < /workspace/train_console.log 2>/dev/null | grep -oE "[0-9]+/19073" | tail -1 | cut -d/ -f1)
[ -z "$step" ] && step=$(grep -oE "^[0-9]+ train" "$LOG" 2>/dev/null | tail -1 | grep -oE "^[0-9]+")
[ -z "$step" ] && step=0
echo "--- progress ---"
echo "step         : $step / 19073  ($(python3 -c "print(f'{$step/19073*100:.1f}')")%)"
tr '\r' '\n' < /workspace/train_console.log 2>/dev/null | tail -1 | sed 's/^/  /'
echo "--- loss trend (downsampled: every ~500 steps, then the latest) ---"
grep -E "^[0-9]+ train .*eta " "$LOG" 2>/dev/null \
  | awk 'NR%50==1{printf "  step %-6s loss %-9s norm %-7s tok/s %-8s eta %s\n",$1,$3,$5,$11,$NF}' | tail -8
grep -E "^[0-9]+ train .*eta " "$LOG" 2>/dev/null | tail -1 \
  | awk '{printf "  step %-6s loss %-9s norm %-7s tok/s %-8s eta %s   <- latest\n",$1,$3,$5,$11,$NF}'
echo "--- val ---"
grep -E "^[0-9]+ val " "$LOG" 2>/dev/null | tail -4 | sed 's/^/  /'
echo "--- GPU saturation (last 60 samples = ~1h) ---"
tail -60 /workspace/gpu_history.log 2>/dev/null | awk -F, '
  {gsub("%","",$2); u=$2+0; s+=u; n++; if(u<min||n==1)min=u; if(u>max)max=u;
   gsub("W","",$4); p+=$4+0}
  END{if(n)printf "  util avg %.1f%%  min %d%%  max %d%%   power avg %.0fW   samples %d\n",s/n,min,max,p/n,n}'
low=$(tail -60 /workspace/gpu_history.log 2>/dev/null | awk -F, '{gsub("%","",$2); if($2+0<80) c++} END{print c+0}')
echo "  samples below 80% util: $low"
# timestamps matter: dips during a restart/compile are expected, dips while the
# trainer is steadily running are not.
[ "${low:-0}" -gt 0 ] && tail -60 /workspace/gpu_history.log 2>/dev/null \
  | awk -F, '{gsub("%","",$2); if($2+0<80) print "    dip at "$1"  util "$2"%  mem "$3}' | tail -8
echo "--- processes ---"
for pat in "train.py --data-root" "pipeline.sh" "preserve_ckpts.sh" "gpu_history.sh"; do
  first_char="${pat:0:1}"; rest="${pat:1}"
  n=$(pgrep -cf "[$first_char]$rest" 2>/dev/null || echo 0)
  printf "  %-24s %s\n" "$pat" "$([ "$n" -gt 0 ] && echo "alive" || echo "NOT RUNNING")"
done
echo "--- checkpoints ---"
ls -1 "$RUN"/checkpoints/ 2>/dev/null | sed 's/^/  /'
echo "  preserved: $(ls -1 /workspace/keep/ 2>/dev/null | tr '\n' ' ')"
echo "--- pipeline ---"
tail -3 /workspace/pipeline.log 2>/dev/null | sed 's/^/  /'
echo "--- ETA ---"
python3 -c "
s=$step;
if s>0:
    rem=19073-s; secs=rem*524288/132000
    print(f'  {rem} steps left -> {secs/3600:.1f}h')
" 2>/dev/null
echo "=========================================================================="
