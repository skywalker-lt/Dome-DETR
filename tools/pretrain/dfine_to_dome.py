"""Rename a D-FINE checkpoint (3 levels: P3-P5) to Dome-DETR key names (4 levels: P2-P5) for `tuning:` initialisation.

D-FINE level i corresponds to Dome level i+1, so per-level modules shift: encoder.input_proj.{0,1,2} -> {1,2,3},
downsample_convs.{0,1} -> {1,2}, pan_blocks.{0,1} -> {1,2}; lateral_convs / fpn_blocks keep their index (both count from
the top level down); backbone, AIFI and decoder keys are unchanged. The solver's tuning loader then copies every tensor
whose name and shape match and skips the rest (thin-neck P2/P3 widths, 4-level cross-attention offsets, class heads).
usage: python dfine_to_dome.py IN.pth OUT.pth [DOME_CONFIG]   (with a config: also reports which tensors will load)"""
import re, sys
import torch

SHIFT = [(r"^encoder\.input_proj\.(\d)\.", "encoder.input_proj.{}."), (r"^encoder\.downsample_convs\.(\d)\.", "encoder.downsample_convs.{}."),
         (r"^encoder\.pan_blocks\.(\d)\.", "encoder.pan_blocks.{}.")]

def rename(k):
    for pat, fmt in SHIFT:
        m = re.match(pat, k)
        if m:
            return fmt.format(int(m.group(1)) + 1) + k[m.end():]
    return k

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd["ema"]["module"] if "ema" in sd else sd["model"]
    out = {rename(k): v for k, v in sd.items()}
    torch.save({"model": out}, dst)
    print(f"wrote {dst}: {len(out)} tensors")
    if len(sys.argv) > 3:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.core import YAMLConfig
        m = YAMLConfig(sys.argv[3], HGNetv2={"pretrained": False}).model
        cur = m.state_dict()
        hit = {k: v for k, v in out.items() if k in cur and cur[k].shape == v.shape}
        n_cur = sum(v.numel() for v in cur.values() if v.is_floating_point())
        n_hit = sum(v.numel() for v in hit.values() if v.is_floating_point())
        groups = {}
        for k in cur:
            g = ".".join(k.split(".")[:2]); h = k in hit
            a = groups.setdefault(g, [0, 0]); a[0] += cur[k].numel(); a[1] += cur[k].numel() if h else 0
        print(f"matched {len(hit)}/{len(cur)} tensors, {100 * n_hit / n_cur:.1f}% of float parameters")
        for g, (t, h) in sorted(groups.items()):
            print(f"  {g:32s} {100 * h / max(t, 1):5.1f}% of {t / 1e6:.3f}M")
