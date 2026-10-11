"""HGNetv2 + DINOv3-distilled ViT-Tiny semantic path for Dome-DETR.

The HGNetv2 backbone is unchanged; a ViT-Tiny (DEIMv2's DINOv3-distilled weights, patch 16) runs on the same image
(ImageNet-normalized, as DEIMv2 trains it) and its intermediate tokens are added into every HGNetv2 output level
through a 1x1 conv + BN whose BN scale starts at zero, so at step 0 the outputs equal plain HGNetv2's. Level mapping
follows DEIMv2 (layers 3, 7, 11 -> strides 8, 16, 32); the stride-4 level takes layer 3 upsampled.

Fusion parameters live under ``sem_fuse`` so optimizer regexes can give them the head learning rate.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .hgnetv2 import HGNetv2
from .vit_tiny import VisionTransformer

__all__ = ["HGNetv2ViT"]


@register()
class HGNetv2ViT(nn.Module):
    def __init__(
        self,
        name="B0",
        return_idx=[0, 1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
        pretrained=True,
        local_model_dir="weight/hgnetv2/",
        out_channels=[64, 256, 512, 1024],
        vit_embed_dim=192,
        vit_num_heads=3,
        vit_layers=[3, 7, 11],
        level_to_vit=[0, 0, 1, 2],
        vit_weights="weight/dinov3/vitt_distill.pt",
        vit_finetune=True,
    ):
        super().__init__()
        self.hgnet = HGNetv2(name=name, return_idx=return_idx, freeze_at=freeze_at, freeze_norm=freeze_norm,
                             use_lab=use_lab, pretrained=pretrained, local_model_dir=local_model_dir)
        self.vit = VisionTransformer(embed_dim=vit_embed_dim, num_heads=vit_num_heads, return_layers=vit_layers)
        if vit_weights:
            if not os.path.exists(vit_weights):
                raise FileNotFoundError(f"ViT-Tiny weights not found: {vit_weights}")
            self.vit._model.load_state_dict(torch.load(vit_weights, map_location="cpu"))
            print(f"Loaded ViT-Tiny weights from {vit_weights}")
        if not vit_finetune:
            self.vit.requires_grad_(False)
        assert len(out_channels) == len(return_idx) == len(level_to_vit)
        self.level_to_vit = list(level_to_vit)
        self.sem_fuse = nn.ModuleList()
        for c in out_channels:
            conv = nn.Conv2d(vit_embed_dim, c, 1, bias=False)
            bn = nn.BatchNorm2d(c)
            nn.init.zeros_(bn.weight)
            self.sem_fuse.append(nn.Sequential(conv, bn))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.patch_size = self.vit.patch_size

    def forward(self, x):
        feats = self.hgnet(x)
        B, _, H, W = x.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        toks = self.vit((x - self.mean) / self.std)
        sem = [t.transpose(1, 2).reshape(B, -1, gh, gw) for t, _ in toks]
        out = []
        for i, f in enumerate(feats):
            s = sem[self.level_to_vit[i]]
            if s.shape[-2:] != f.shape[-2:]:
                s = F.interpolate(s, size=f.shape[-2:], mode="bilinear", align_corners=False)
            out.append(f + self.sem_fuse[i](s))
        return out
