"""GFLOPs and parameters of thin (N1-E-style) variants of Dome-DETR S/M/L, untrained, on the count_gf.py basis.
FLOPs of a Dome model depend on the data only through W (active MWAS windows) and Q (decoder queries). Both are forced
here (W: a mask with exactly W active windows; Q: min_num_select = max_num_select = Q), so one image suffices. Counted at
(W, Q) in {(40, 600), (50, 600), (40, 700)} -> GF = s + a W + b Q, then evaluated at each size's measured full-test means.
usage: python count_variants.py SIZE VARIANT  (one model per process: YAMLConfig carries sections over between builds)"""
import json, sys, os, copy
import torch
from torch.utils.flop_counter import FlopCounterMode
REPO = "/data/tmp/dome/repo"; sys.path.insert(0, REPO); os.chdir(REPO)
torch.set_num_threads(8)
from src.core import YAMLConfig  # noqa: E402
from src.zoo.dome.fold import fold_exact  # noqa: E402
from src.zoo.dome import hybrid_encoder as he  # noqa: E402

FORCE_W = {"k": 40}
def forced_filter(self, feat, init_thresh=0.05, step=0.01):  # exactly k active 10x10 windows on the P3 grid
    m = torch.zeros_like(feat, dtype=torch.bool); B, _, H, W = feat.shape; ws = 10; nw = W // ws
    for i in range(FORCE_W["k"]):
        r, c = divmod(i, nw); m[:, :, r * ws + ws // 2, c * ws + ws // 2] = True
    return m
he.HybridEncoder.adaptive_defe_filter = forced_filter

THIN = {"level_dims": [64, 128, 256, 256]}
VAR = {
    "dense": ({}, {}),
    "thin": (dict(THIN), {"feat_channels": [64, 128, 256, 256]}),
    "n1e": (dict(THIN, defe_width=64, defe_head_width=32), {"feat_channels": [64, 128, 256, 256]}),
    "nodens": (dict(THIN, defe_width=64, defe_head_width=32, use_defe=False, use_mwas=False), {"feat_channels": [64, 128, 256, 256]}),
    "x3": ({"level_dims": [64, 128, 128, 128], "hidden_dim": 128, "dim_feedforward": 512, "defe_width": 64, "defe_head_width": 32},
           {"feat_channels": [64, 128, 128, 128]}),
}
x = torch.rand(1, 3, 800, 800)

def count(size, enc, dec, W, Q):
    FORCE_W["k"] = W
    over = {"HGNetv2": {"pretrained": False}, "HybridEncoder": dict(enc), "DomeTransformer": dict(dec, min_num_select=Q, max_num_select=Q)}
    cfg = YAMLConfig(f"configs/dome/Dome-{size}-AITOD.yml", **copy.deepcopy(over))
    for sec, kv in over.items(): cfg.yaml_cfg.setdefault(sec, {}).update(kv)
    m = cfg.model.eval(); params = sum(p.numel() for p in m.parameters()) / 1e6
    fold_exact(m)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 1e9, params

size, v = sys.argv[1], sys.argv[2]
enc, dec = VAR[v]
g00, p = count(size, enc, dec, 40, 600); g10, _ = count(size, enc, dec, 50, 600); g01, _ = count(size, enc, dec, 40, 700)
a, b = (g10 - g00) / 10, (g01 - g00) / 100; s0 = g00 - 40 * a - 600 * b
print(json.dumps({"key": f"{size}_{v}", "params_M": round(p, 3), "s": s0, "a_per_window": a, "b_per_query": b}), flush=True)
