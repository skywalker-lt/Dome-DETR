#!/usr/bin/env bash
# Resumable single-GPU training launcher with backups to the shared /training_data volume.
# usage: tools/pod/launch.sh CONFIG RUN_NAME [SEED]
#  - run dir: the config's output_dir on local disk (e.g. /root/runs/<name>)
#  - resumes from the local last.pth, else from the backup copy, else starts fresh
#  - every 10 completed epochs (and at the end) copies last.pth, best_stg*.pth, fit_state.json, log.txt,
#    train.out to $BACKUP_ROOT/<name>/  (checkpoints only; no caches, no datasets)
set -uo pipefail
CFG=$1; NAME=$2; SEED=${3:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
BACKUP_ROOT=${BACKUP_ROOT:-/training_data/dome-thin-twin}
RUN=$($PY - "$REPO/$CFG" <<'PYEOF'
import sys, os; sys.path.insert(0, os.getcwd())
from src.core import YAMLConfig
print(YAMLConfig(sys.argv[1]).output_dir)
PYEOF
)
RUN=${RUN##*$'\n'}
BK="$BACKUP_ROOT/$NAME"
mkdir -p "$RUN" "$BK"
cd "$REPO"
if [ ! -f "$RUN/last.pth" ] && [ -f "$BK/last.pth" ]; then
  echo "[launch] restoring $BK -> $RUN"; cp -n "$BK"/*.pth "$BK"/fit_state.json "$BK"/log.txt "$RUN/" 2>/dev/null
fi
RESUME=(); [ -f "$RUN/last.pth" ] && RESUME=(-r "$RUN/last.pth") && echo "[launch] resuming from $RUN/last.pth"

backup() {
  for f in last.pth best_stg1.pth best_stg2.pth fit_state.json log.txt train.out; do
    [ -f "$RUN/$f" ] && cp "$RUN/$f" "$BK/$f.tmp" && mv "$BK/$f.tmp" "$BK/$f"
  done
  echo "[launch] $(date -u +%FT%TZ) backed up epoch $1 to $BK" >> "$RUN/backup.log"
}
watch_backups() {
  local done=-1
  while sleep 300; do
    ep=$(tail -n 1 "$RUN/log.txt" 2>/dev/null | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['epoch'])" 2>/dev/null) || continue
    # first poll only records the decade; afterwards back up whenever a new block of 10 epochs completes
    if [ $done -lt 0 ]; then done=$(( (ep + 1) / 10 )); elif [ $(( (ep + 1) / 10 )) -gt $done ]; then done=$(( (ep + 1) / 10 )); backup "$ep"; fi
  done
}
watch_backups & WATCH=$!
export PYTHONUNBUFFERED=1
# The pod container is CPU-quota limited (27 cores); torch defaults to one OpenMP thread per host core (256),
# which spins and gets throttled. A few threads per run is enough for the small CPU-side ops.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
$PY train.py -c "$CFG" --seed "$SEED" "${RESUME[@]}" >> "$RUN/train.out" 2>&1
RC=$?
kill $WATCH 2>/dev/null
backup final
echo "[launch] train.py exited $RC" | tee -a "$RUN/train.out"
exit $RC
