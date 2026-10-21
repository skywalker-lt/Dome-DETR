# YOLO-style necks for Dome-DETR

`src/zoo/dome/yolo_encoder.py` builds the Dome-DETR encoder (neck) from a YOLO-style layer list, so blocks can be added,
removed, re-wired or resized in YAML instead of code. Select it in a run config:

```yaml
DOME:
  encoder: YoloEncoder
YoloEncoder:
  model_yaml: configs/dome/yolo/dome-s-thin-neck.yaml
```

Rows are `[from, repeats, module, args, released-name]`. Layers 0-3 are the backbone outputs (P2-P5); rows are numbered
from 4; negative `from` is relative. Input channels are inferred. `outputs` lists the four layers passed to the decoder
(set `DomeTransformer.feat_channels` to their channels). A trailing dict in `args` is passed as keyword arguments;
`YoloEncoder.defe_overrides` applies keyword arguments to every DeFEMWAS row (the FP-1 mixin uses it).

| block | inputs | args |
|---|---|---|
| InputProj | 1 | `[c2]` 1x1 conv + BN |
| AIFI | 1 | `[nhead, ffn, dropout, act, layers]` transformer with 2D sin-cos positions |
| ConvNormFuse | 1 | `[c2, k, s]` conv + BN (fusable) |
| UpConcat | 2 | `[]` upsample the first to the second's size (bilinear, align_corners), concat |
| Concat | n | `[]` |
| RepNCSPELAN4 | 1 | `[c2, c3, c4, n]` |
| SCDown | 1 | `[c2, k, s]` |
| DeFEMWAS | 2 | `[window, defe_width, defe_head_width, mwas_ffn]` + kwargs; density on input 1, window attention on input 2 |
| Identity | 1 | `[]` |

`dome-s-dense-neck.yaml`, `dome-s-thin-neck.yaml` and `dome-s-n1e-neck.yaml` are the released / thin / N1-E necks.
With the released-name field, `convert_hybrid_state_dict` loads HybridEncoder checkpoints; `tools/pod/verify_yolo_encoder.py`
checks that the YAML neck gives the same outputs (all three: bit-identical eval outputs and training losses).
Note: the module creation order differs from HybridEncoder, so a from-scratch run with the same seed starts from a
different random initialisation; compare YAML-neck variants against a YAML-neck control.
