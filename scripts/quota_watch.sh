#!/usr/bin/env bash
# Alert while there is still room, instead of after torch.save is truncated.
# The volume quota is invisible to df (MooseFS reports cluster-wide free space),
# so the only reliable probe is to actually try writing.
while true; do
  if ! dd if=/dev/zero of=/workspace/.quotaprobe bs=1M count=600 >/dev/null 2>&1; then
    printf '%s | !!! QUOTA: cannot write 600MB — a 1.9GB checkpoint WILL truncate\n' \
      "$(date -u '+%H:%M:%SZ')" | tee -a /workspace/watchdog.log
    rm -f /workspace/.quotaprobe; exit 12
  fi
  rm -f /workspace/.quotaprobe
  sleep 120
done
