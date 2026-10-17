"""Exact inference-time rewrites of a trained Dome-DETR model (same function up to fp32 rounding, fewer FLOPs).

- cut0  DeFE: the 1x1 conv and the 2x2 average pool commute (both linear, pool weights sum to 1) -> pool first.
- cut0c DeFE density head Conv3x3(w->h) -> bilinear x4 -> Conv1x1(h->1) -> sigmoid: nothing nonlinear between the two
        convs and bilinear upsampling is linear with weights summing to 1, so the 1x1 folds into the 3x3:
        Conv3x3(w->1) -> bilinear x4 -> sigmoid.
- cut0b decoder: on levels whose input projection is a 1x1 conv + BN (thin necks), fold it into enc_output.proj for the
        query-scoring path. The projected memory is still built (it is the deformable-attention value). Exact only in
        eval mode (BN running statistics), so the decoder uses the folded path only when not training.

Call after model.to(device).eval(). Verified against the unfolded model with tools/pod/verify_fold.py."""
import torch
import torch.nn as nn


@torch.no_grad()
def fold_exact(model, cuts=("cut0", "cut0c", "cut0b")):
    applied = []
    enc = getattr(model, "encoder", None)
    defe = getattr(enc, "DeFE", None)
    if defe is not None and "cut0" in cuts:
        conv, pool = defe.conv1[0], defe.conv1[1]
        if isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1) and isinstance(pool, nn.AvgPool2d):
            defe.conv1 = nn.Sequential(pool, conv)
            applied.append("cut0")
    if defe is not None and "cut0c" in cuts:
        h = defe.density_head
        c3, up, c1, act = h[0], h[1], h[2], h[3]
        if isinstance(c1, nn.Conv2d) and c1.kernel_size == (1, 1) and c1.out_channels == 1:
            w1 = c1.weight.double()[:, :, 0, 0]  # 1 x h
            head = nn.Conv2d(c3.in_channels, 1, c3.kernel_size, padding=c3.padding).to(c3.weight.device)
            head.weight.copy_(torch.einsum("oc,cikl->oikl", w1, c3.weight.double()).to(c3.weight.dtype))
            head.bias.copy_((w1 @ c3.bias.double() + c1.bias.double()).to(c3.bias.dtype))
            defe.density_head = nn.Sequential(head, up, act)
            applied.append("cut0c")
    dec = getattr(model, "decoder", None)
    if dec is not None and "cut0b" in cuts:
        P = dec.enc_output.proj
        Wl, bl = P.weight.double(), P.bias.double()
        folds = []
        for ip in dec.input_proj:
            conv = getattr(ip, "conv", None)
            if conv is None or conv.kernel_size != (1, 1) or conv.stride != (1, 1):
                folds.append(None)
                continue
            bn = ip.norm
            s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
            Wc = conv.weight.double()[:, :, 0, 0] * s[:, None]
            bc = bn.bias.double() - bn.running_mean.double() * s
            folds.append(((Wl @ Wc).to(P.weight.dtype), (Wl @ bc + bl).to(P.bias.dtype)))
        if any(f is not None for f in folds):
            dec.folded_enc_proj = folds
            applied.append("cut0b")
    return applied
