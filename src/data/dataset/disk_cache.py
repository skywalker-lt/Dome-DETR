"""Decoded-image disk cache (like Ultralytics ``cache=disk``): each image is stored once as an uncompressed uint8
HWC ``.npy`` on a LOCAL disk, so training skips PNG decoding and never reads images over a network volume.

The cache directory must not sit on a network filesystem (MooseFS ``/data``, geesefs ``/training_data``, NFS, ...):
``assert_local_dir`` refuses it, because random reads of ~2 MB files from those mounts stall the dataloader.
"""
import os

import numpy as np
from PIL import Image

NETWORK_FS = ("fuse", "nfs", "cifs", "smb", "ceph", "glusterfs", "lustre", "9p", "geesefs", "moosefs", "mfs")
FORBIDDEN_PREFIXES = ("/data", "/training_data")


def _mount_of(path):
    path = os.path.realpath(path)
    best, fstype = "", ""
    with open("/proc/mounts") as f:
        for line in f:
            parts = line.split()
            mnt, typ = parts[1], parts[2]
            if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                best, fstype = mnt, typ
    return best, fstype


def assert_local_dir(path):
    """Raise if ``path`` resolves to /data, /training_data or any network filesystem."""
    real = os.path.realpath(path)
    for p in FORBIDDEN_PREFIXES:
        if real == p or real.startswith(p + "/"):
            raise ValueError(f"cache_dir {path} -> {real} is on network volume {p}; use a local disk")
    mnt, typ = _mount_of(real)
    if any(k in typ.lower() for k in NETWORK_FS):
        raise ValueError(f"cache_dir {path} -> {real} is on {typ} mount {mnt}; use a local disk")
    return real


def cache_path(cache_dir, file_name):
    return os.path.join(cache_dir, os.path.splitext(file_name)[0] + ".npy")


def write_cached(cache_dir, file_name, src_path):
    dst = cache_path(cache_dir, file_name)
    if os.path.exists(dst):
        return dst
    arr = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    tmp = dst + f".tmp{os.getpid()}"
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, dst)
    return dst


def load_cached(cache_dir, file_name):
    return Image.fromarray(np.load(cache_path(cache_dir, file_name)))
