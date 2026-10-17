"""Evaluate a released Dome-DETR AI-TOD-v2 checkpoint on a COCO-format split (CPU) and capture its intermediates.

Outputs in <out>/: results.json (COCO detections, category ids as in the given GT json), metrics.txt (the repo's
AI-TOD evaluator summary), per_image.csv (gt count, DeFE count regression, active MWAS windows, query count, detections),
density.npz (per image: DeFE density map at stride 4 and its window-pooled map, fp16), window_mask.npz (MWAS window masks).
usage: python run_eval.py <config.yml> <ckpt.pth> <img_dir> <gt.json> <out_dir> [--limit N] [--threads T]
       [--fold] [--set Section.key=value ...]
--fold applies the exact inference-time rewrites (src/zoo/dome/fold.py); --set overrides config entries, e.g.
--set DomeTransformer.paqi_density_fix=True --set DomeTransformer.max_num_select=600 --set HybridEncoder.window_threshold=0.2"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("ckpt"); ap.add_argument("img_dir"); ap.add_argument("gt"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--device", default="cpu"); ap.add_argument("--no-eval", action="store_true"); ap.add_argument("--no-maps", action="store_true"); ap.add_argument("--repo", default="/data/tmp/dome/repo"); ap.add_argument("--fold", action="store_true"); ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
REPO = a.repo; sys.path.insert(0, REPO); os.chdir(REPO)
torch.set_num_threads(a.threads)
from src.core import YAMLConfig  # noqa: E402

import yaml  # noqa: E402
over = {"HGNetv2": {"pretrained": False}}
for kv in a.set:
    key, val = kv.split("=", 1); sec, k = key.split(".", 1)
    over.setdefault(sec, {})[k] = yaml.safe_load(val)
cfg = YAMLConfig(a.config, **over)
cfg.yaml_cfg.setdefault("HGNetv2", {})["pretrained"] = False
for sec, kv in over.items():  # make sure the overrides reach the built modules
    cfg.yaml_cfg.setdefault(sec, {}).update(kv)
print("overrides", over, flush=True)
model, post = cfg.model, cfg.postprocessor
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
state = ck["ema"]["module"] if "ema" in ck else ck["model"]
missing, unexpected = model.load_state_dict(state, strict=False)
print("loaded", a.ckpt, "missing", len(missing), "unexpected", len(unexpected), flush=True)
dev = torch.device(a.device); model.to(dev).eval(); post.eval()
if a.fold:
    from src.zoo.dome.fold import fold_exact  # noqa: E402
    print("folded", fold_exact(model), flush=True)

cap = {}
def enc_hook(mod, inp, out):
    d = out.get("defe", {}) if isinstance(out, dict) else {}
    cap["density"] = d.get("density_map"); cap["pooled"] = d.get("density_map_pooled")
    cap["wmask"] = d.get("defe_window_mask"); cap["reg"] = d.get("reg_value")
model.encoder.register_forward_hook(enc_hook)

gt = json.load(open(a.gt))
gt_count = {}
for an in gt["annotations"]:
    gt_count[an["image_id"]] = gt_count.get(an["image_id"], 0) + 1
images = gt["images"][: a.limit] if a.limit else gt["images"]
# the released models predict labels in the official AI-TOD-v2 id space; detect the offset to this json from the label range
os.makedirs(a.out, exist_ok=True)
from faster_coco_eval import COCO  # noqa: E402
from src.data.dataset.coco_eval_aitod import AitodCocoEvaluator  # noqa: E402
import contextlib, io  # noqa: E402
cg = COCO(a.gt); ev = AitodCocoEvaluator(cg, ["bbox"])
gt_cat_ids = {c["id"] for c in gt["categories"]}
det_fh = open(f"{a.out}/results.jsonl", "w"); n_det = 0; chunk = 0
rows, dens, pooled, wmasks = [], {}, {}, {}
def flush_maps():
    global dens, pooled, wmasks, chunk
    if dens or wmasks:
        np.savez_compressed(f"{a.out}/density_{chunk:03d}.npz", **{f"d_{k}": v for k, v in dens.items()}, **{f"p_{k}": v for k, v in pooled.items()})
        np.savez_compressed(f"{a.out}/window_mask_{chunk:03d}.npz", **wmasks)
        chunk += 1
    dens, pooled, wmasks = {}, {}, {}
t0 = time.time()
with torch.no_grad():
    for k, im in enumerate(images):
        img = Image.open(os.path.join(a.img_dir, im["file_name"])).convert("RGB")
        w, h = img.size
        x = torch.from_numpy(np.asarray(img.resize((800, 800), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        out = model(x.to(dev))
        res = {k: v.cpu() for k, v in post(out, torch.tensor([[w, h]], dtype=torch.float32, device=dev))[0].items()}
        lab, box, sc = res["labels"].numpy(), res["boxes"].numpy(), res["scores"].numpy()
        keep = np.isin(lab, list(gt_cat_ids))  # the extra background-index class is not a COCO category
        if not a.no_eval: ev.update({im["id"]: {"boxes": torch.from_numpy(box[keep]), "scores": torch.from_numpy(sc[keep]), "labels": torch.from_numpy(lab[keep]).long()}})
        for l, b, s_ in zip(lab[keep], box[keep], sc[keep]):
            det_fh.write(json.dumps({"image_id": im["id"], "category_id": int(l), "bbox": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)], "score": round(float(s_), 5)}) + "\n")
        n_det += int(keep.sum())
        nq = out.get("batch_queries_num", [len(lab)])
        nq = int(nq[0]) if hasattr(nq, "__len__") else int(nq)
        wm = cap.get("wmask")
        wm = wm.cpu() if wm is not None else None
        nwin = int(wm.sum()) if wm is not None else -1
        ntot = int(wm.numel()) if wm is not None else -1
        reg = cap.get("reg")
        regv = float(reg.flatten()[0].cpu()) if reg is not None and reg.numel() else float("nan")
        rows.append([im["id"], im["file_name"], gt_count.get(im["id"], 0), regv, nwin, ntot, nq, len(lab), int((sc > 0.3).sum())])
        if cap.get("density") is not None and not a.no_maps:
            dens[str(im["id"])] = cap["density"][0, 0].cpu().numpy().astype(np.float16)
            pooled[str(im["id"])] = cap["pooled"][0, 0].cpu().numpy().astype(np.float16)
        if wm is not None and not a.no_maps:
            wmasks[str(im["id"])] = wm[0].numpy().astype(np.uint8)
        if (k + 1) % 250 == 0:
            flush_maps()
        if k % 50 == 0:
            import resource
            print(f"RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB at {k}", flush=True)
            print(f"{k}/{len(images)} {time.time() - t0:.0f}s gt {rows[-1][2]} queries {nq} windows {nwin}/{ntot}", flush=True)

flush_maps(); det_fh.close()
with open(f"{a.out}/per_image.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["image_id", "file", "gt_count", "defe_reg_value", "mwas_active_windows", "mwas_total_windows", "queries", "detections", "detections_score_gt_0.3"]); wr.writerows(rows)
print(f"inference done {time.time() - t0:.0f}s, {n_det} detections", flush=True)
if a.no_eval:
    sys.exit(0)
ev.synchronize_between_processes(); ev.accumulate()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ev.summarize()
txt = buf.getvalue(); print(txt)
open(f"{a.out}/metrics.txt", "w").write(txt)
