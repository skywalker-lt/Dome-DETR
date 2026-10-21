"""Check that a YOLO-style neck (YoloEncoder) reproduces the released HybridEncoder on a trained checkpoint.

Builds both models (each in its own process, to avoid config-registry carry-over), loads the checkpoint (renamed for
the YoloEncoder), and compares on real test images: eval-mode density map, windows, queries, final logits and boxes;
and one training-mode forward + criterion with the same seed. usage:
python tools/pod/verify_yolo_encoder.py RELEASED_CFG YOLO_CFG CKPT N"""
import json, os, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r'''
import json, os, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, REPO); os.chdir(REPO); torch.set_num_threads(8)
from src.core import YAMLConfig
from src.zoo.dome.yolo_encoder import YoloEncoder, convert_hybrid_state_dict
cfg = YAMLConfig(CFG, HGNetv2={"pretrained": False}); m, crit = cfg.model, cfg.criterion
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["ema"]["module"]
if isinstance(m.encoder, YoloEncoder):
    sd = convert_hybrid_state_dict(sd, m.encoder)
m.load_state_dict(sd, strict=True)
G = json.load(open("/data/datasets/AI-TOD-v2/aitodv2_anns/aitodv2_test.json")); by = {}
for a in G["annotations"]: by.setdefault(a["image_id"], []).append(a)
ims = [im for im in G["images"] if 30 <= len(by.get(im["id"], [])) <= 400][:N]
res = {"params": sum(p.numel() for p in m.parameters()), "eval": []}
m.eval()
xs, tg = [], []
for im in ims:
    img = Image.open(os.path.join("/data/datasets/AI-TOD-v2/images/test", im["file_name"])).convert("RGB"); w, h = img.size
    x = torch.from_numpy(np.asarray(img.resize((800, 800)), dtype=np.float32) / 255.).permute(2, 0, 1)
    with torch.no_grad():
        o = m(x[None])
    res["eval"].append({"density": o["defe"]["density_map"].flatten()[::97].tolist(), "windows": int(o["defe"]["defe_window_mask"].sum()),
                        "queries": int(o["batch_queries_num"][0]), "logits": o["pred_logits"][0].flatten()[::13].tolist(), "boxes": o["pred_boxes"][0].flatten()[::7].tolist()})
    xs.append(x); b = torch.tensor([a["bbox"] for a in by[im["id"]]], dtype=torch.float32)
    tg.append({"boxes": torch.stack([(b[:, 0] + b[:, 2] / 2) / w, (b[:, 1] + b[:, 3] / 2) / h, b[:, 2] / w, b[:, 3] / h], 1),
               "labels": torch.tensor([a["category_id"] - 1 for a in by[im["id"]]]), "image_id": torch.tensor([im["id"]]),
               "orig_size": torch.tensor([w, h]), "size": torch.tensor([800, 800]),
               "area": torch.tensor([a["area"] for a in by[im["id"]]], dtype=torch.float32), "iscrowd": torch.zeros(len(by[im["id"]]), dtype=torch.long)})
m.train(); crit.train(); torch.manual_seed(0)
o = m(torch.stack(xs[:2]), targets=[{k: v.clone() for k, v in t.items()} for t in tg[:2]])
L = crit(o, tg[:2], epoch=150, step=0, global_step=0, epoch_step=1752)
res["train_losses"] = {k: float(v) for k, v in sorted(L.items())}
print("RESULT" + json.dumps(res))
'''

def run(cfg, ckpt, n):
    code = f"REPO={REPO!r}; CFG={cfg!r}; CKPT={ckpt!r}; N={n}\n" + CHILD
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        sys.exit(out.stdout[-3000:] + out.stderr[-3000:])
    return json.loads(line[0][6:])

if __name__ == "__main__":
    import numpy as np
    rel_cfg, yolo_cfg, ckpt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    a, b = run(rel_cfg, ckpt, n), run(yolo_cfg, ckpt, n)
    print("params", a["params"], b["params"])
    for i, (ea, eb) in enumerate(zip(a["eval"], b["eval"])):
        d = {k: float(np.max(np.abs(np.array(ea[k]) - np.array(eb[k])))) if len(ea[k]) == len(eb[k]) else "shape"
             for k in ("density", "logits", "boxes")}
        print(f"img {i}: windows {ea['windows']}/{eb['windows']} queries {ea['queries']}/{eb['queries']} max diff {d}")
    la, lb = a["train_losses"], b["train_losses"]
    print("train losses max |diff|:", max(abs(la[k] - lb[k]) for k in la), "over", len(la), "terms; total", sum(la.values()), sum(lb.values()))
