#!/usr/bin/env bash
# OOM-loop breaker (user 2026-10-26: decide scheduling to maximize efficiency). Every 60 s: if a running run sits on
# the same training step at oom-retry attempt >= 8, pause the OTHER run that holds the most GPU memory (park it, hold
# the queue), wait until the stuck run gets past that step (or 20 min), then resume the paused run (exact resume from
# last.pth) and release the queue. Logs to /root/queue/unstick.log.
Q=/root/queue
cfg_of() { grep -h "^$1:" $Q/running.txt $Q/fired.txt | tail -1; }
while sleep 60; do
  [ -f $Q/unstick.busy ] && continue
  for spec in $(cat $Q/running.txt $Q/fired.txt); do
    n=${spec%%:*}; R=/root/runs/dome_s_$n; f=$R/train.out
    { [ -f $R/FINISHED ] || [ -f $R/PARKED ]; } && continue
    last=$(tail -n 1 $f 2>/dev/null)
    a=$(echo "$last" | grep -oE "^\[oom-retry\] epoch [0-9]+ step [0-9]+ attempt [0-9]+" | awk "{print \$3, \$5, \$7}")
    [ -z "$a" ] && continue
    set -- $a; ep=$1; st=$2; att=${3%%/*}; [ "$att" -lt 8 ] && continue
    # pick the other unfinished run holding the most GPU memory
    best=""; bestm=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F", " -v p=$p "\$1==p{print \$2}")
      o=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -oE "/root/runs/dome_s_[a-z0-9_]+" | head -1)
      [ -z "$o" ] && o=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -oE "Dome-S-[a-z0-9-]+-AITOD-pod.yml" | head -1)
      for s2 in $(cat $Q/running.txt $Q/fired.txt); do n2=${s2%%:*}; c2=$(echo $s2 | cut -d: -f2)
        { [ "$o" = "${c2##*/}" ] || [ "$o" = "/root/runs/dome_s_$n2" ]; } || continue
        [ "$n2" = "$n" ] && continue
        [ "${m:-0}" -gt "$bestm" ] && { best=$n2; bestm=$m; }
      done
    done
    [ -z "$best" ] && continue
    touch $Q/unstick.busy $Q/STOP
    echo "$(date -u +%FT%TZ) $n stuck at epoch $ep step $st (attempt $att); pausing $best (${bestm} MiB)"
    echo "paused by unstick.sh for $n" > /root/runs/dome_s_$best/PARKED; tmux kill-session -t =$best
    t0=$(date +%s)
    while sleep 30; do
      tail -n 1 $f | grep -q "epoch $ep step $st attempt" || { tail -n 3 $f | grep -q "step $st attempt" || break; }
      [ $(( $(date +%s) - t0 )) -ge 1200 ] && break
    done
    echo "$(date -u +%FT%TZ) $n moved on (or 20 min); resuming $best"
    rm -f /root/runs/dome_s_$best/PARKED; bash /root/restart_runs.sh "$(cfg_of $best)"
    rm -f $Q/STOP $Q/unstick.busy
    break
  done
done
