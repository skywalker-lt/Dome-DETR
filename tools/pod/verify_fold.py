"""Check that src/zoo/dome/fold.py's exact rewrites leave a trained model's output unchanged, and count the FLOPs saved.

Runs the unfolded and the folded model on the first N test images at 800x800 and compares density map, MWAS windows,
query count, final logits and boxes. usage: python tools/pod/verify_fold.py CONFIG CKPT IMG_DIR N [--device cuda]"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.core import YAMLConfig  # noqa: E402
from src.zoo.dome.fold import fold_exact  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("ckpt"); ap.add_argument("img_dir"); ap.add_argument("n", type=int)
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--size", type=int, default=800)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.chdir(REPO)
dev = torch.device(a.device)


def load():
    m = YAMLConfig(a.config, HGNetv2={"pretrained": False}).model
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd["ema"]["module"] if "ema" in sd else sd["model"]
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not missing, missing
    return m.to(dev).eval()


ref, cut = load(), load()
applied = fold_exact(cut)
print("applied", applied, flush=True)


def run(m, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        o = m(x)
    return o, fc.get_total_flops() / 1e9


rows = []
for f in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[: a.n]:
    x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((a.size, a.size)), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None].to(dev)
    o0, g0 = run(ref, x)
    o1, g1 = run(cut, x)
    q = [int(o0["batch_queries_num"][0]), int(o1["batch_queries_num"][0])]
    wdw = [int(o["defe"]["defe_window_mask"].sum()) if "defe" in o else -1 for o in (o0, o1)]
    same = o0["pred_logits"].shape == o1["pred_logits"].shape
    # order-independent: rounding-level differences may reorder queries at top-k, so match every folded query to its
    # nearest unfolded query (L1 over box + logits) and report the worst match
    z0 = torch.cat([o0["pred_boxes"][0], o0["pred_logits"][0]], -1)[: q[0]]
    z1 = torch.cat([o1["pred_boxes"][0], o1["pred_logits"][0]], -1)[: q[1]]
    nn_d = torch.cdist(z1.double(), z0.double(), p=1).min(1).values
    r = dict(img=os.path.basename(f), gf_ref=round(g0, 3), gf_cut=round(g1, 3), queries=q, windows=wdw,
             d_density=float((o0["defe"]["density_map"] - o1["defe"]["density_map"]).abs().max()) if "defe" in o0 else 0.0,
             d_logits=float((o0["pred_logits"] - o1["pred_logits"]).abs().max()) if same else None,
             d_boxes=float((o0["pred_boxes"] - o1["pred_boxes"]).abs().max()) if same else None,
             matched_max_l1=float(nn_d.max()), unmatched_1e3=int((nn_d > 1e-3).sum()))
    rows.append(r)
    print(json.dumps(r), flush=True)
summary = dict(applied=applied, n=len(rows), gf_ref_mean=float(np.mean([r["gf_ref"] for r in rows])),
               gf_cut_mean=float(np.mean([r["gf_cut"] for r in rows])),
               identical_counts=all(r["queries"][0] == r["queries"][1] and r["windows"][0] == r["windows"][1] for r in rows),
               max_d_logits=max((r["d_logits"] if r["d_logits"] is not None else float("inf")) for r in rows),
               max_d_boxes=max((r["d_boxes"] if r["d_boxes"] is not None else float("inf")) for r in rows),
               max_d_density=max(r["d_density"] for r in rows),
               matched_max_l1=max(r["matched_max_l1"] for r in rows), unmatched_1e3=sum(r["unmatched_1e3"] for r in rows))
print("SUMMARY", json.dumps(summary), flush=True)
