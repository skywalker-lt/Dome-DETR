"""Dome-DETR encoder (neck) built from a YOLO-style layer list.

The YAML describes the neck the way an Ultralytics model file does: one row per layer, `[from, repeats, module, args]`,
input channels inferred from the `from` layers. Layers 0..N-1 are the backbone outputs (P2..P5 for Dome), rows are
numbered from N on, negative `from` indices are relative to the current row. `outputs` lists the layers passed to the
decoder (strides as in `strides`). An optional fifth field names the matching module of the released HybridEncoder, so
existing checkpoints load through `convert_hybrid_state_dict`.

Blocks: InputProj (1x1 conv + BN), AIFI (transformer on one level with 2D sin-cos positions), ConvNormFuse (conv + BN
[+ act], fusable), UpConcat (bilinear-upsample the first input to the second's size and concatenate), Concat,
RepNCSPELAN4, SCDown, Identity, and DeFEMWAS (density estimation on the first input, window attention on the second;
writes the density outputs that the decoder and the criterion use). See configs/dome/yolo/ for the released Dome-S
necks in this form.

    # [from, repeats, module, args, (released name)]
    - [0, 1, InputProj, [64], input_proj.0]
    - [[4, 5], 1, DeFEMWAS, [10, 256, 128]]
    - [[-1, 6], 1, UpConcat, []]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...core import register
from .defe import GaussHeatmapGenerator, LiteDeFE
from .get_roi_features import TransformerEncoder, TransformerEncoderLayer, WindowProcessor
from .hybrid_encoder import ConvNormLayer_fuse, HybridEncoder, RepNCSPELAN4, SCDown

__all__ = ["YoloEncoder", "convert_hybrid_state_dict", "load_layer_yaml"]


class InputProj(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__()
        self.add_module("conv", nn.Conv2d(c1, c2, kernel_size=1, bias=False))
        self.add_module("norm", nn.BatchNorm2d(c2))


class AIFI(nn.Module):
    """Single-level transformer encoder (D-FINE / Dome AIFI) with 2D sin-cos positions built for the actual size."""

    def __init__(self, c1, nhead=8, dim_feedforward=1024, dropout=0.0, act="gelu", num_layers=1, temperature=10000.0):
        super().__init__()
        layer = TransformerEncoderLayer(c1, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=act)
        self.enc = TransformerEncoder(layer, num_layers)
        self.c, self.temperature = c1, temperature

    def forward(self, x):
        h, w = x.shape[2:]
        src = x.flatten(2).permute(0, 2, 1)
        pos = HybridEncoder.build_2d_sincos_position_embedding(w, h, self.c, self.temperature).to(src.device)
        mem = self.enc(src, pos_embed=pos)
        return mem.permute(0, 2, 1).reshape(-1, self.c, h, w).contiguous()


class ConvNormFuse(ConvNormLayer_fuse):
    def __init__(self, c1, c2, k=1, s=1, act=None):
        super().__init__(c1, c2, k, s, act=act)


class UpConcat(nn.Module):
    """[a, b] -> concat(bilinear_upsample(a to b's size, align_corners=True), b)."""

    def forward(self, xs):
        a, b = xs
        a = F.interpolate(a, size=(b.shape[2], b.shape[3]), mode="bilinear", align_corners=True)
        return torch.concat([a, b], dim=1)


class Concat(nn.Module):
    def forward(self, xs):
        return torch.concat(xs, dim=1)


class DeFEMWAS(nn.Module):
    """[p2, p3] -> p3 enhanced by window attention on the windows the density map selects.

    Writes ctx['defe'] (density_map, pooled map, reg_value, window mask, GT density map) as HybridEncoder does."""

    def __init__(self, c_p2, c_p3, window_size=10, defe_width=256, defe_head_width=128, mwas_ffn=1024,
                 eval_per_image_norm=False, train_per_image_norm=False, window_threshold=0.05, use_mwas=True):
        super().__init__()
        self.DeFE = LiteDeFE(in_channels=c_p2, width=defe_width, head_width=defe_head_width,
                             eval_per_image_norm=eval_per_image_norm, train_per_image_norm=train_per_image_norm)
        self.use_mwas = use_mwas
        if use_mwas:
            self.mwas_processor = WindowProcessor(embed_dim=c_p3, dim_feedforward=mwas_ffn, num_layers=1)
        self.window_size, self.window_threshold, self.c_p3 = window_size, window_threshold, c_p3

    def forward(self, xs, ctx):
        p2, p3 = xs
        density, reg_value = self.DeFE(p2)
        W, H = p3.shape[2:]
        d = {"reg_value": reg_value, "density_map": density}
        pooled = F.adaptive_max_pool2d(density, (H // self.window_size, W // self.window_size))
        d["defe_feature"] = density
        d["density_map_pooled"] = pooled
        if self.use_mwas:
            filt = HybridEncoder.adaptive_defe_filter(  # uses no encoder state
                None, F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=True),
                init_thresh=self.window_threshold).float()
            pos = HybridEncoder.build_2d_sincos_position_embedding(W, H, embed_dim=self.c_p3).permute(0, 2, 1).view(-1, H, W).to(p3.device)
            p3, mask = self.mwas_processor(p3, filt, self.window_size, pos)
            d["defe_window_mask"] = mask
        d["gt_density_map"] = []
        img, targets = ctx["img_inputs"], ctx["targets"]
        if targets is not None:
            B, C, Hi, Wi = img.shape
            gen = GaussHeatmapGenerator(img_size=(Hi, Wi))
            for b in range(B):
                boxes = targets[b]["boxes"]
                if not self.training:  # as HybridEncoder: eval targets are xyxy in pixels
                    boxes[:, 0] = boxes[:, 0] / Wi
                    boxes[:, 1] = boxes[:, 1] / Hi
                    boxes[:, 2] = boxes[:, 2] / Wi
                    boxes[:, 3] = boxes[:, 3] / Hi
                    boxes = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)
                d["gt_density_map"].append(gen(boxes))
            d["gt_density_map"] = torch.stack(d["gt_density_map"]).to(img.device)
        ctx["defe"] = d
        return p3


BLOCKS = {"InputProj": InputProj, "AIFI": AIFI, "ConvNormFuse": ConvNormFuse, "UpConcat": UpConcat, "Concat": Concat,
          "RepNCSPELAN4": RepNCSPELAN4, "SCDown": SCDown, "DeFEMWAS": DeFEMWAS, "Identity": nn.Identity}


def _build(name, cin, args):
    """Module and output channels for one row; cin is the list of input channels. A trailing dict in args is passed
    as keyword arguments."""
    kw = {}
    if args and isinstance(args[-1], dict):
        args, kw = list(args[:-1]), dict(args[-1])
    if name in ("UpConcat", "Concat"):
        return BLOCKS[name](), sum(cin)
    if name == "Identity":
        return nn.Identity(), cin[0]
    if name == "DeFEMWAS":
        return DeFEMWAS(cin[0], cin[1], *args, **kw), cin[1]
    if name == "AIFI":
        return AIFI(cin[0], *args, **kw), cin[0]
    if name == "RepNCSPELAN4":  # [c2, c3, c4, n]
        return RepNCSPELAN4(cin[0], *args, **kw), args[0]
    if name in ("InputProj", "ConvNormFuse", "SCDown"):  # first arg = output channels
        return BLOCKS[name](cin[0], *args, **kw), args[0]
    raise KeyError(f"unknown block {name}; known: {sorted(BLOCKS)}")


def load_layer_yaml(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


@register()
class YoloEncoder(nn.Module):
    """Drop-in replacement for HybridEncoder whose neck is read from a YOLO-style layer list (`model_yaml`)."""

    def __init__(self, model_yaml, in_channels=None, feat_strides=None, defe_overrides=None):
        """defe_overrides: keyword args applied to every DeFEMWAS row (e.g. the FP-1 protocol's
        eval_per_image_norm), so config mixins reach the YAML neck."""
        super().__init__()
        defe_overrides = dict(defe_overrides or {})
        spec = load_layer_yaml(model_yaml)
        self.in_channels = list(in_channels if in_channels is not None else spec["inputs"])
        assert list(spec.get("inputs", self.in_channels)) == self.in_channels, (spec.get("inputs"), self.in_channels)
        n_in = len(self.in_channels)
        ch = list(self.in_channels)
        self.froms, self.names, self.legacy, layers = [], [], [], []
        for i, row in enumerate(spec["encoder"]):
            f, n, name, args = row[:4]
            args = list(args or [])
            if name == "DeFEMWAS" and defe_overrides:
                kw = dict(args.pop()) if args and isinstance(args[-1], dict) else {}
                args = args + [{**kw, **defe_overrides}]
            idx = n_in + i
            fl = [f] if isinstance(f, int) else list(f)
            fl = [j if j >= 0 else idx + j for j in fl]
            assert all(0 <= j < idx for j in fl), f"row {idx}: from {f} refers to a later layer"
            cin = [ch[j] for j in fl]
            m, c2 = _build(name, cin, args)
            if n > 1:
                assert name not in ("UpConcat", "Concat", "DeFEMWAS") and c2 == cin[0], f"row {idx}: repeats need c_in == c_out"
                m = nn.Sequential(m, *[_build(name, [c2], args)[0] for _ in range(n - 1)])
            layers.append(m)
            self.froms.append(fl if not isinstance(f, int) else fl[0])
            self.names.append(name)
            self.legacy.append(row[4] if len(row) > 4 else None)
            ch.append(c2)
        self.layers = nn.ModuleList(layers)
        self.outputs = list(spec["outputs"])
        self.out_channels = [ch[j] for j in self.outputs]
        self.out_strides = list(spec.get("strides", feat_strides or []))
        self.n_in = n_in

    def forward(self, feats, img_inputs, targets=None):
        assert len(feats) == self.n_in
        ctx = {"img_inputs": img_inputs, "targets": targets}
        y = list(feats)
        for m, f, name in zip(self.layers, self.froms, self.names):
            x = y[f] if isinstance(f, int) else [y[j] for j in f]
            y.append(m(x, ctx) if name == "DeFEMWAS" else m(x))
        out = {"img_inputs": img_inputs, "feats": [y[j] for j in self.outputs]}
        if "defe" in ctx:
            out["defe"] = ctx["defe"]
        return out


def convert_hybrid_state_dict(sd, encoder: YoloEncoder, prefix="encoder."):
    """Rename a released HybridEncoder checkpoint's encoder keys to a YoloEncoder built from a YAML whose rows carry
    the released module names (5th field). Keys outside the encoder are returned unchanged."""
    rename = []
    for i, (name, leg) in enumerate(zip(encoder.names, encoder.legacy)):
        if leg is None:
            continue
        tgt = f"{prefix}layers.{i}."
        if name == "DeFEMWAS":
            rename += [(f"{prefix}DeFE.", tgt + "DeFE."), (f"{prefix}mwas_processor.", tgt + "mwas_processor.")]
        elif name == "AIFI":
            rename.append((f"{prefix}{leg}.", tgt + "enc."))
        else:
            rename.append((f"{prefix}{leg}.", tgt))
    out, used = {}, set()
    for k, v in sd.items():
        for a, b in rename:
            if k.startswith(a):
                out[b + k[len(a):]] = v
                used.add(a)
                break
        else:
            out[k] = v
    missing = [a for a, _ in rename if a not in used]
    assert not missing, f"released names not found in the checkpoint: {missing}"
    return out
