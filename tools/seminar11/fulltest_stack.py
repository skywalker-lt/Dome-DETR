"""d1 (round 2, point 3, with d2): ONE batch-1 full-test job for the inference-only cuts and their stack.
For every test image and every variant: detections (results.jsonl in run_eval.py's format), per-image W (MWAS windows),
Q (decoder queries), B (queries kept after decoder layer 1; = Q when pruning is off), and on every --count-every'th image the
counted GFLOPs (FlopCounterMode, verify_fold.py convention) so the full-test mean/worst follow by an exact-form fit in (W, Q, B).
Variants (YAML neck, so that d1's rewrites apply; the YAML neck is bit-identical to HybridEncoder, README):
  E    fold_exact (control; must reproduce the existing final_rescore detections)
  E2   E + d1's five rewrites (rep, cut0e, cut0f, cut0d, cut0g)          -> must equal E (exact; W, Q identical)
  EP   E + d2's frozen-context pruning (thr 0.02, min 100, 8 classes)    -> d2's P1
  E2P  E2 + the same pruning                                              -> must equal EP (W, Q, B identical)
Needs a repo copy with d2/patches/query_prune.patch applied (--repo) and d1_folds.py on the path (--d1).
usage: python fulltest_stack.py --repo R --d1 D1DIR --ckpt CKPT --name NAME --img-dir DIR --gt GT.json --out OUT
       [--variants E,E2,EP,E2P] [--count-every 10] [--limit 0] [--device cuda] [--threads 4]"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--d1", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--img-dir", required=True); ap.add_argument("--gt", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--variants", default="E,E2,EP,E2P")
ap.add_argument("--count-every", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sys.path.insert(0, a.repo); sys.path.insert(0, a.d1); os.chdir(a.repo)
from PIL import Image  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from src.core import YAMLConfig  # noqa: E402
from src.zoo.dome.fold import fold_exact  # noqa: E402
from src.zoo.dome.yolo_encoder import convert_hybrid_state_dict  # noqa: E402
import d1_folds  # noqa: E402

CFG = "configs/dome/pod/Dome-S-n1e-yolo-AITOD-pod.yml"
P1 = {"query_prune_thr": 0.02, "query_prune_min": 100, "query_prune_classes": 8, "query_prune_frozen": True}
OFF = {"query_prune_thr": None, "query_prune_frozen": False}
FOLDS = ["rep", "cut0e", "cut0f", "cut0d", "cut0g"]
dev = torch.device(a.device)
sd0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd0 = sd0["ema"]["module"] if "ema" in sd0 else sd0["model"]


def build(v):
    prune, d1 = v in ("EP", "E2P"), v in ("E2", "E2P")
    # pass the pruning keys every time: YAMLConfig keeps included sections in shared dicts (carry-over between builds)
    over = {"HGNetv2": {"pretrained": False}, "DomeTransformer": dict(P1) if prune else dict(OFF)}
    cfg = YAMLConfig(CFG, **over)
    for sec, kv in over.items():
        cfg.yaml_cfg.setdefault(sec, {}).update(kv)
    m, post = cfg.model, cfg.postprocessor
    assert not m.load_state_dict(convert_hybrid_state_dict(dict(sd0), m.encoder), strict=False)[0]
    m.to(dev).eval(); post.eval()
    applied = fold_exact(m)
    if d1:
        applied += d1_folds.apply(m, FOLDS)
    assert (m.decoder.decoder.query_prune_thr == 0.02) == prune, "config carry-over"
    return m, post, applied


variants = a.variants.split(",")
M = {v: build(v) for v in variants}
print({v: M[v][2] for v in variants}, flush=True)
gt = json.load(open(a.gt)); cats = {c["id"] for c in gt["categories"]}
images = gt["images"][: a.limit] if a.limit else gt["images"]
od = os.path.join(a.out, a.name); os.makedirs(od, exist_ok=True)
fh = {v: open(f"{od}/{v}_results.jsonl", "w") for v in variants}
rows, t0 = [], time.time()
with torch.no_grad():
    for k, im in enumerate(images):
        img = Image.open(os.path.join(a.img_dir, im["file_name"])).convert("RGB"); w, h = img.size
        x = torch.from_numpy(np.asarray(img.resize((800, 800), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        row = [im["id"], im["file_name"]]
        for v in variants:
            m, post, _ = M[v]
            count = a.count_every > 0 and k % a.count_every == 0
            if count:
                with FlopCounterMode(display=False) as fc:
                    out = m(x)
                gf = fc.get_total_flops() / 1e9
            else:
                out, gf = m(x), float("nan")
            res = {kk: vv.cpu() for kk, vv in post(out, torch.tensor([[w, h]], dtype=torch.float32, device=dev))[0].items()}
            lab, box, sc = res["labels"].numpy(), res["boxes"].numpy(), res["scores"].numpy()
            keep = np.isin(lab, list(cats))
            for l, b, s_ in zip(lab[keep], box[keep], sc[keep]):
                fh[v].write(json.dumps({"image_id": im["id"], "category_id": int(l), "bbox": [round(float(b[0]), 2), round(float(b[1]), 2),
                                        round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)], "score": round(float(s_), 5)}) + "\n")
            W = int(out["defe"]["defe_window_mask"].sum()); Q = int(out["batch_queries_num"][0]); B = int(out["pred_logits"].shape[1])
            row += [W, Q, B, round(gf, 4)]
        rows.append(row)
        if k % 200 == 0:
            print(f"{k}/{len(images)} {time.time() - t0:.0f}s {row[2:]}", flush=True)
for f in fh.values():
    f.close()
with open(f"{od}/per_image.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["image_id", "file"] + [f"{v}_{c}" for v in variants for c in ("W", "Q", "B", "GF")]); wr.writerows(rows)
print(f"done {time.time() - t0:.0f}s", flush=True)
