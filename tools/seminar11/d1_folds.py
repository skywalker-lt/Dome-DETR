"""d1: further exact inference-time rewrites of a trained Dome-DETR with a YOLO-style neck (YoloEncoder), on top of
src/zoo/dome/fold.py (cut0, cut0c, cut0b). Same function up to fp32 rounding, fewer counted FLOPs. Apply after fold_exact,
model in eval mode. apply(model, ["rep", "cut0e", "cut0f", "cut0d"]).

rep    RepVGG: every VGGBlock (3x3+BN || 1x1+BN) -> one 3x3 conv (VGGBlock.convert_to_deploy, as D-FINE's deploy mode).
cut0e  FPN: RepNCSPELAN4.cv1 is a 1x1 conv + BN (linear) + SiLU on concat(upsample(a), b). Bilinear upsampling
       (align_corners=True, weights sum to 1) commutes with a 1x1 linear map, so the a-half of cv1 is applied at the low
       resolution before upsampling: SiLU(up(Wa' a) + Wb' b + t). Saves 3/4 of the a-half of cv1 at P4, P3 and P2.
       (The upsample then interpolates c3 = 2*c_a channels instead of c_a; interpolation is not counted by the convention.)
cut0f  P2 input projection (1x1 conv + BN, linear, row 4) feeds only DeFE (whose first op after cut0 is avg-pool then
       1x1 conv) and the b-half of the P2 FPN block's cv1: fold it into both and drop it. Requires cut0 and cut0e.
cut0d  Decoder value projection (thin levels: 1x1 conv + BN, C_in -> 256) is linear, and deformable attention samples the
       projected map bilinearly with zero padding: gs(A x + b) = A gs(x) + b gs(1). So sample the C_in-channel map (plus a
       ones channel) per head and project the samples (Q x layers x heads x points per level) instead of every position.
       Chosen per image and level only when it is cheaper (Q < h*w/12 with 3 layers, 8 heads, 4 points, head dim 32).
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn_affine(bn):
    s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
    return s, bn.bias.double() - bn.running_mean.double() * s


class FoldedUpELAN(nn.Module):
    """Replaces [UpConcat -> RepNCSPELAN4]: input (a, b); output RepNCSPELAN4(concat(up(a), b))."""

    def __init__(self, elan, c_a):
        super().__init__()
        cv1 = elan.cv1
        assert not hasattr(cv1, "conv_bn_fused") and cv1.conv.kernel_size == (1, 1) and cv1.conv.bias is None
        s, t = _bn_affine(cv1.norm)
        Wt = cv1.conv.weight.double()[:, :, 0, 0] * s[:, None]  # c3 x (c_a + c_b)
        self.wa = nn.Conv2d(c_a, Wt.shape[0], 1, bias=False)
        self.wb = nn.Conv2d(Wt.shape[1] - c_a, Wt.shape[0], 1, bias=True)
        dt = cv1.conv.weight.dtype
        with torch.no_grad():
            self.wa.weight.copy_(Wt[:, :c_a, None, None].to(dt))
            self.wb.weight.copy_(Wt[:, c_a:, None, None].to(dt))
            self.wb.bias.copy_(t.to(dt))
        self.act = cv1.act
        self.cv2, self.cv3, self.cv4, self.c = elan.cv2, elan.cv3, elan.cv4, elan.c
        self.wa = self.wa.to(cv1.conv.weight.device); self.wb = self.wb.to(cv1.conv.weight.device)

    def fold_input_affine(self, A, a):
        """b was produced by y = A x + a (A: c_b x c_x, a: c_b); make wb act on x directly."""
        W = self.wb.weight.double()[:, :, 0, 0]
        with torch.no_grad():
            nb = nn.Conv2d(A.shape[1], W.shape[0], 1, bias=True).to(self.wb.weight.device)
            nb.weight.copy_((W @ A)[:, :, None, None].to(self.wb.weight.dtype))
            nb.bias.copy_((W @ a + self.wb.bias.double()).to(self.wb.bias.dtype))
        self.wb = nb

    def forward(self, ab):
        a, b = ab
        za = self.wa(a)
        za = F.interpolate(za, size=(b.shape[2], b.shape[3]), mode="bilinear", align_corners=True)
        y = self.act(za + self.wb(b))
        y = list(y.split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class Pack(nn.Module):
    def forward(self, xs):
        return tuple(xs)


def _cut0e(enc):
    done = []
    for i, name in enumerate(enc.names):
        if name != "UpConcat":
            continue
        consumers = [j for j, f in enumerate(enc.froms) if (f == enc.n_in + i) or (isinstance(f, list) and enc.n_in + i in f)]
        if len(consumers) != 1 or enc.names[consumers[0]] != "RepNCSPELAN4" or consumers[0] != i + 1:
            continue
        fa = enc.froms[i][0]
        c_a = (enc.in_channels + [None] * 0)[fa] if fa < enc.n_in else _out_ch(enc, fa - enc.n_in)
        enc.layers[i] = Pack()
        enc.layers[i + 1] = FoldedUpELAN(enc.layers[i + 1], c_a)
        enc.names[i] = "Pack"; enc.names[i + 1] = "FoldedUpELAN"
        done.append(enc.n_in + i + 1)
    return done


def _out_ch(enc, li):
    m = enc.layers[li]
    if hasattr(m, "cv4"):
        return m.cv4.conv.out_channels if hasattr(m.cv4, "conv") else m.cv4.conv_bn_fused.out_channels
    if hasattr(m, "conv"):
        return m.conv.out_channels
    if hasattr(m, "c"):
        return m.c
    raise ValueError(f"out channels of layer {li}")


def _cut0f(enc):
    from src.zoo.dome.defe import LiteDeFE
    li = 0  # row 4 = layers[0], the P2 InputProj
    assert enc.names[li] == "InputProj" and enc.froms[li] == 0
    row = enc.n_in + li
    consumers = [j for j, f in enumerate(enc.froms) if (f == row) or (isinstance(f, list) and row in f)]
    conv, bn = enc.layers[li].conv, enc.layers[li].norm
    s, t = _bn_affine(bn)
    A = conv.weight.double()[:, :, 0, 0] * s[:, None]; a = t
    for j in consumers:
        nm = enc.names[j]
        if nm == "DeFEMWAS":
            assert enc.froms[j][0] == row
            defe = enc.layers[j].DeFE
            assert isinstance(defe, LiteDeFE) and isinstance(defe.conv1[0], nn.AvgPool2d), "needs cut0 (pool first)"
            c = defe.conv1[1]
            W = c.weight.double()[:, :, 0, 0]
            with torch.no_grad():
                nc = nn.Conv2d(A.shape[1], W.shape[0], 1, bias=True).to(c.weight.device)
                nc.weight.copy_((W @ A)[:, :, None, None].to(c.weight.dtype))
                nc.bias.copy_((W @ a + c.bias.double()).to(c.bias.dtype))
            defe.conv1 = nn.Sequential(defe.conv1[0], nc)
        elif nm == "Pack":
            assert enc.froms[j][1] == row and enc.names[j + 1] == "FoldedUpELAN", "needs cut0e"
            enc.layers[j + 1].fold_input_affine(A, a)
        else:
            raise ValueError(f"unexpected consumer {nm} of the P2 input projection")
    enc.layers[li] = nn.Identity(); enc.names[li] = "Identity"
    return [row]


def _cut0d(dec):
    """Decoder: project thin-level values after bilinear sampling (see module doc)."""
    lv = []
    for l, ip in enumerate(dec.input_proj):
        conv = getattr(ip, "conv", None)
        if conv is None or conv.kernel_size != (1, 1) or conv.stride != (1, 1):
            lv.append(None); continue
        s, t = _bn_affine(ip.norm)
        lv.append(((conv.weight.double()[:, :, 0, 0] * s[:, None]).to(conv.weight.dtype), t.to(conv.weight.dtype)))
    if not any(x is not None for x in lv):
        return []
    dec._d1_lv = lv
    nh = dec.nhead
    tdec = dec.decoder
    n_layers = len(tdec.layers)
    pts = [layer.cross_attn.num_points_list for layer in tdec.layers]

    def get_enc_input(self, feats):
        proj, flat, shapes = [], [], []
        self._d1_raw = {}
        for i, f in enumerate(feats):
            _, _, h, w = f.shape
            shapes.append([h, w])
            if self._d1_lv[i] is not None:
                self._d1_raw[i] = f
                flat.append(f.new_zeros(f.shape[0], h * w, self.hidden_dim))  # never read: folded scoring path + cut0d values
                proj.append(None)
            else:
                p = self.input_proj[i](f); proj.append(p); flat.append(p.flatten(2).permute(0, 2, 1))
        assert self.num_levels == len(feats)
        return proj, torch.concat(flat, 1), shapes

    dec._get_encoder_input = types.MethodType(get_enc_input, dec)
    assert dec.folded_enc_proj is not None, "cut0d needs cut0b (folded enc_output) so the zero memory is never read"
    orig_forward = tdec.forward
    orig_value_op = tdec.value_op

    def tdec_forward(self, target, *args, **kw):
        self._d1_Q = target.shape[1]
        return orig_forward(target, *args, **kw)

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        vals = list(orig_value_op(memory, value_proj, value_scale, memory_mask, memory_spatial_shapes))
        Q = self._d1_Q
        for l, fl in enumerate(dec._d1_lv):
            if fl is None:
                continue
            x = dec._d1_raw[l]
            h, w = x.shape[2:]
            if Q * n_layers * nh * pts[0][l] * 32 < h * w * 256:  # sampled-point projection is cheaper
                vals[l] = ("d1fold", x, fl[0], fl[1])
            else:
                p = dec.input_proj[l](x).flatten(2).permute(0, 2, 1)  # [bs, hw, 256]
                vals[l] = p.reshape(p.shape[0], p.shape[1], nh, -1).permute(0, 2, 3, 1)
        return vals

    tdec.forward = types.MethodType(tdec_forward, tdec)
    tdec.value_op = types.MethodType(value_op, tdec)

    def core(value, value_spatial_shapes, sampling_locations, attention_weights, num_points_list, method="default"):
        assert method == "default"
        bs, Len_q = sampling_locations.shape[:2]
        n_head = sampling_locations.shape[2]
        grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)  # [bs*nh, Lq, P, 2]
        grids = grids.split(num_points_list, dim=-2)
        out = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            v = value[level]
            g = grids[level]
            if isinstance(v, tuple):
                _, x, A, b = v
                C = x.shape[1]
                xa = torch.cat([x, x.new_ones(x.shape[0], 1, h, w)], 1)  # ones channel: in-bound weight sum
                xa = xa.unsqueeze(1).expand(bs, n_head, C + 1, h, w).reshape(bs * n_head, C + 1, h, w)
                s = F.grid_sample(xa, g, mode="bilinear", padding_mode="zeros", align_corners=False)  # [bs*nh, C+1, Lq, P]
                s = s.view(bs, n_head, C + 1, Len_q, -1)
                Ah = A.view(n_head, -1, C)  # head h uses output channels [32h, 32h+32)
                sv = torch.einsum("bhcqp,hdc->bhdqp", s[:, :, :C], Ah) + b.view(n_head, -1)[None, :, :, None, None] * s[:, :, C:]
                out.append(sv.reshape(bs * n_head, -1, Len_q, sv.shape[-1]))
            else:
                c = v.shape[2]
                vl = v.reshape(bs * n_head, c, h, w)
                out.append(F.grid_sample(vl, g, mode="bilinear", padding_mode="zeros", align_corners=False))
        aw = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
        o = (torch.concat(out, dim=-1) * aw).sum(-1).reshape(bs, -1, Len_q)
        return o.permute(0, 2, 1)

    for layer in tdec.layers:
        layer.cross_attn.ms_deformable_attn_core = core
    return [f"cut0d(levels {[l for l, x in enumerate(lv) if x is not None]})"]


class _LazyEncMemory:
    """cut0g: the folded encoder output LN(W_f x + b_f) (+ invalid-anchor rows = LN(proj.bias)) without materialising
    256-d rows for every anchor. Scores of folded levels: score head after LayerNorm is
    (G y)/sqrt(var(y) + eps) + c with y = W_f x + b_f, G = W_s diag(gamma) P (P: centring), var(y) = |P y|^2 / D; so
    numerator = M x + m and var = |R x + d|^2 + e, with B = P W_f / sqrt(D) = Q R (QR), x0 = least-squares solution of
    B x0 = P b_f / sqrt(D), d = R x0, e = |residual|^2. Rows are built only for the selected (top-k) anchors."""

    def __init__(self, dec, memory, feats, spatial_shapes, valid_mask):
        self.dec, self.memory, self.feats, self.shapes, self.valid = dec, memory, feats, spatial_shapes, valid_mask.bool()

    def logits(self):
        dec = self.dec; proj, norm, sh = dec.enc_output.proj, dec.enc_output.norm, dec.enc_score_head.lin
        out, n0 = [], 0
        for lvl, (h, w) in enumerate(self.shapes):
            n = h * w
            g = dec._d1_g[lvl]
            if g is None:
                out.append(sh(norm(proj(self.memory[:, n0:n0 + n]))))
            else:
                x = self.feats[lvl].flatten(2).permute(0, 2, 1)
                M, m, R, d, e, c = g
                u = x @ R.t() + d
                var = (u * u).sum(-1, keepdim=True) + e
                out.append((x @ M.t() + m) / torch.sqrt(var + norm.eps) + c)
            n0 += n
        lg = torch.cat(out, 1)
        inv = sh(norm(proj.bias))  # rows of invalid anchors (as the released path: proj applied to a zeroed memory)
        return torch.where(self.valid, lg, inv.expand_as(lg))

    def rows(self, ind):
        dec = self.dec; proj, norm = dec.enc_output.proj, dec.enc_output.norm
        bounds, n0 = [], 0
        for (h, w) in self.shapes:
            bounds.append((n0, n0 + h * w)); n0 += h * w
        bs, k = ind.shape
        y = torch.empty(bs, k, proj.out_features, dtype=self.memory.dtype, device=self.memory.device)
        for lvl, (a0, a1) in enumerate(bounds):
            sel = (ind >= a0) & (ind < a1)
            if not sel.any():
                continue
            b_idx, j_idx = sel.nonzero(as_tuple=True)
            loc = ind[b_idx, j_idx] - a0
            fold = dec.folded_enc_proj[lvl] if lvl < len(dec.folded_enc_proj) else None
            if fold is None:
                y[b_idx, j_idx] = proj(self.memory[b_idx, a0 + loc])
            else:
                x = self.feats[lvl].flatten(2).permute(0, 2, 1)[b_idx, loc]
                y[b_idx, j_idx] = x @ fold[0].t() + fold[1]
        v = self.valid[0][ind]  # valid_mask is [1, N, 1] -> [bs, k, 1]
        y = torch.where(v, y, proj.bias.expand_as(y))
        return norm(y)


class _ScoreHead(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return x.logits() if isinstance(x, _LazyEncMemory) else self.lin(x)


def _cut0g(dec):
    assert dec.folded_enc_proj is not None, "cut0g needs cut0b"
    proj, norm = dec.enc_output.proj, dec.enc_output.norm
    D = proj.out_features
    Ws, bs_ = dec.enc_score_head.weight.double(), dec.enc_score_head.bias.double()
    gam, bet = norm.weight.double(), norm.bias.double()
    P = torch.eye(D, dtype=torch.float64) - 1.0 / D
    G = Ws * gam[None, :] @ P
    c = (Ws @ bet + bs_)
    gs, used = [], []
    for lvl, fold in enumerate(dec.folded_enc_proj):
        if fold is None:
            gs.append(None); continue
        Wf, bf = fold[0].double(), fold[1].double()
        cin = Wf.shape[1]
        B = P @ Wf / D ** 0.5; b0 = P @ bf / D ** 0.5
        Qm, R = torch.linalg.qr(B)
        x0 = torch.linalg.lstsq(B, b0[:, None]).solution[:, 0]
        e = float(((B @ x0 - b0) ** 2).sum())
        dt = fold[0].dtype
        gs.append(((G @ Wf).to(dt), (G @ bf).to(dt), R.to(dt), (R @ x0).to(dt), e, c.to(dt)))
        used.append(lvl)
    dec._d1_g = gs
    dec.enc_score_head = _ScoreHead(dec.enc_score_head)
    orig_fold_out = dec._folded_enc_output
    dec._folded_enc_output = types.MethodType(lambda self, memory, feats, shapes, valid: _LazyEncMemory(self, memory, feats, shapes, valid), dec)
    orig_sel = dec._select_topk

    def select_topk(memory, logits, anchors, topk):
        if not isinstance(memory, _LazyEncMemory):
            return orig_sel(memory, logits, anchors, topk)
        assert dec.query_select_method == "default"
        _, ind = torch.topk(logits.max(-1).values, topk, dim=-1)
        ta = anchors.gather(1, ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
        tl = logits.gather(1, ind.unsqueeze(-1).repeat(1, 1, logits.shape[-1]))
        return memory.rows(ind), tl, ta

    dec._select_topk = select_topk
    return [f"cut0g(levels {used})"]


@torch.no_grad()
def apply(model, cuts):
    from src.zoo.dome.hybrid_encoder import VGGBlock
    from src.zoo.dome.yolo_encoder import YoloEncoder
    applied = []
    enc = model.encoder
    if "rep" in cuts:
        n = 0
        for m in model.modules():
            if isinstance(m, VGGBlock) and not hasattr(m, "conv"):
                m.convert_to_deploy(); n += 1
        applied.append(f"rep({n})")
    if "cut0e" in cuts:
        assert isinstance(enc, YoloEncoder)
        applied.append(f"cut0e(rows {_cut0e(enc)})")
    if "cut0f" in cuts:
        applied.append(f"cut0f(rows {_cut0f(enc)})")
    if "cut0d" in cuts:
        applied += _cut0d(model.decoder)
    if "cut0g" in cuts:
        applied += _cut0g(model.decoder)
    return applied
