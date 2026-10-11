#!/usr/bin/env bash
# One-time pod setup for AI-TOD-v2: annotations to local disk, decoded-image cache on local disk.
# Reads images once from the /data volume; writes nothing to /data or /training_data.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=${SRC:-/data/datasets/AI-TOD-v2}
DST=${DST:-/root/aitod}
PY=${PY:-python}
mkdir -p "$DST/annotations"
cp -n "$SRC"/aitodv2_anns/aitodv2_{trainval,test}.json "$DST/annotations/"
$PY "$REPO/tools/pod/build_cache.py" "$DST/annotations/aitodv2_trainval.json" "$DST/cache/trainval" "$SRC/images/train" "$SRC/images/val" --workers ${WORKERS:-64}
$PY "$REPO/tools/pod/build_cache.py" "$DST/annotations/aitodv2_test.json" "$DST/cache/test" "$SRC/images/test" --workers ${WORKERS:-64}
du -sh "$DST/cache"/*
