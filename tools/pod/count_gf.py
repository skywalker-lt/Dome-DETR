"""Counted GFLOPs and parameters of a trained Dome-DETR checkpoint, same basis for every model: exact folds applied
(src/zoo/dome/fold.py), batch 1 at the model's eval size, torch FlopCounterMode (verify_fold.py convention), every
--every'th AI-TOD-v2 test image (each model at its own data-dependent MWAS windows / decoder queries).
usage: python tools/pod/count_gf.py CONFIG CKPT IMG_DIR GT.json OUT.json [--every 10] [--device cuda]"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("ckpt"); ap.add_argument("img_dir"); ap.add_argument("gt"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=10); ap.add_argument("--device", default="cuda")
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
a = ap.parse_args()
sys.path.insert(0, a.repo); os.chdir(a.repo)
from src.core import YAMLConfig  # noqa: E402
from src.zoo.dome.fold import fold_exact  # noqa: E402

cfg = YAMLConfig(a.config, HGNetv2={"pretrained": False})
cfg.yaml_cfg.setdefault("HGNetv2", {})["pretrained"] = False
model = cfg.model
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"], strict=False)
params = sum(p.numel() for p in model.parameters())
dev = torch.device(a.device); model.to(dev).eval()
folds = fold_exact(model)
S = int((cfg.yaml_cfg.get("eval_spatial_size") or [800, 800])[0])
images = json.load(open(a.gt))["images"][:: a.every]
gf, t0 = [], time.time()
with torch.no_grad():
    for im in images:
        img = Image.open(os.path.join(a.img_dir, im["file_name"])).convert("RGB")
        x = torch.from_numpy(np.asarray(img.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        with FlopCounterMode(display=False) as fc:
            model(x)
        gf.append(fc.get_total_flops() / 1e9)
r = {"config": a.config, "ckpt": a.ckpt, "missing": len(missing), "unexpected": len(unexpected), "params_M": params / 1e6,
     "folds": folds, "size": S, "images": len(gf), "gf_mean": float(np.mean(gf)), "gf_p50": float(np.median(gf)),
     "gf_max": float(np.max(gf)), "seconds": round(time.time() - t0)}
json.dump(r, open(a.out, "w"), indent=1); print(json.dumps(r), flush=True)
