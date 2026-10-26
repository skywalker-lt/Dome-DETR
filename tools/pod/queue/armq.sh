#!/usr/bin/env bash
# Run queue for pod 3 (runs in tmux session "armq"). Every 5 min: if fewer than MAX runs are unfinished and the GPU
# has at least MIN_FREE_GB free and no run was fired in the last GAP_MIN minutes, fire the first line of queue.txt
# (name:config:seed) through restart_runs.sh (tools/pod/launch.sh: resumable, backups to /training_data) and move
# it to fired.txt. A first line "WAIT" holds the queue (branch not decided yet). Edit queue.txt at any time.
Q=/root/queue; MAX=${MAX:-3}; MIN_FREE_GB=${MIN_FREE_GB:-30}; GAP_MIN=${GAP_MIN:-20}
touch $Q/fired.txt
while sleep 300; do
  [ -f $Q/STOP ] && { echo "$(date -u +%FT%TZ) STOP file present; idle"; continue; }
  next=$(grep -v '^#' $Q/queue.txt 2>/dev/null | grep -v '^\s*$' | head -1)
  [ -z "$next" ] && continue
  [ "$next" = WAIT ] && continue
  active=0
  for n in $(cut -d: -f1 $Q/running.txt $Q/fired.txt); do
    d=/root/runs/dome_s_$n; { [ -f $d/FINISHED ] || [ -f $d/PARKED ]; } || active=$((active + 1))
  done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); free=$((free / 1024))
  last=$(stat -c %Y $Q/fired.txt); age=$(( ($(date +%s) - last) / 60 ))
  [ -s $Q/fired.txt ] || age=999
  if [ $active -lt $MAX ] && [ $free -ge $MIN_FREE_GB ] && [ $age -ge $GAP_MIN ]; then
    echo "$(date -u +%FT%TZ) firing $next (active $active, free ${free} GB)"
    rm -f /root/runs/dome_s_${next%%:*}/PARKED; bash /root/restart_runs.sh "$next"
    echo "$next" >> $Q/fired.txt
    grep -vxF "$next" $Q/queue.txt > $Q/queue.tmp; mv $Q/queue.tmp $Q/queue.txt
  fi
done
