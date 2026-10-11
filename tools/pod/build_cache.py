"""Build the local decoded-image cache for one COCO-style annotation file (see src/data/dataset/disk_cache.py).

usage: python tools/pod/build_cache.py ANN_JSON CACHE_DIR SRC_DIR [SRC_DIR ...] [--workers 64]
Each image is looked up by file_name in the SRC_DIRs (read once, e.g. from the /data volume) and written as
CACHE_DIR/<stem>.npy. CACHE_DIR must be on a local disk; /data, /training_data and network mounts are refused.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.dataset.disk_cache import assert_local_dir, cache_path, write_cached  # noqa: E402


def _one(job):
    cache_dir, fn, srcs = job
    for d in srcs:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            write_cached(cache_dir, fn, p)
            return 0
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ann")
    ap.add_argument("cache_dir")
    ap.add_argument("src_dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    cache_dir = assert_local_dir(a.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    names = [im["file_name"] for im in json.load(open(a.ann))["images"]]
    todo = [n for n in names if not os.path.exists(cache_path(cache_dir, n))]
    print(f"{len(names)} images, {len(todo)} to cache into {cache_dir}", flush=True)
    t = time.time()
    with Pool(a.workers) as pool:
        missing = sum(pool.imap_unordered(_one, [(cache_dir, n, a.src_dirs) for n in todo], chunksize=16))
    print(f"done in {time.time() - t:.0f} s; source not found for {missing} images", flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
