"""Training-only distillation from a frozen Dome-DETR teacher (0 GFLOPs at inference).

Student and teacher share the anchor grid (same strides and input size), so two dense targets line up one to one:
- encoder class logits per anchor (the scores that pick the decoder queries): soft-target BCE, summed over classes and
  averaged over valid anchors, minus the teacher's own entropy (i.e. a per-anchor KL; 0 when the student matches);
- the DeFE density map (both normalised the same way within the batch): mean squared error.

Configured by a top-level `distill:` entry: teacher_config, teacher_ckpt, w_enc, w_density. The teacher is built with
its architecture pinned explicitly and loaded strictly, so a config leak from the student cannot change it silently."""
import torch
import torch.nn.functional as F

from ..core import YAMLConfig

# the teacher (released dense Dome-S) must not inherit student settings that its own yaml leaves at defaults
TEACHER_PINS = {
    "HGNetv2": {"pretrained": False},
    "HybridEncoder": {"level_dims": None, "defe_width": 256, "defe_head_width": 128, "defe_eval_per_image_norm": False,
                      "defe_train_per_image_norm": False, "window_threshold": 0.05},
    "DomeTransformer": {"paqi_density_fix": False, "paqi_iou_offset": 0.0},
}


def _unwrap(m):
    return getattr(m, "module", m)


class TeacherDistiller:
    def __init__(self, teacher_config, teacher_ckpt, device, w_enc=1.0, w_density=1.0):
        cfg = YAMLConfig(teacher_config, **TEACHER_PINS)
        for sec, kv in TEACHER_PINS.items():
            cfg.yaml_cfg.setdefault(sec, {}).update(kv)
        teacher = cfg.model
        sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        sd = sd["ema"]["module"] if "ema" in sd else sd["model"]
        teacher.load_state_dict(sd, strict=True)
        self.teacher = teacher.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        _unwrap(self.teacher).decoder.export_enc_logits = True
        self.w_enc, self.w_density = float(w_enc), float(w_density)
        n = sum(p.numel() for p in self.teacher.parameters())
        print(f"[distill] teacher {teacher_config} ({n / 1e6:.2f}M params) from {teacher_ckpt}; w_enc {w_enc} w_density {w_density}",
              flush=True)

    def attach(self, student):
        _unwrap(student).decoder.export_enc_logits = True

    @torch.no_grad()
    def _teacher_targets(self, samples):
        out = self.teacher(samples)
        t_logits, valid = _unwrap(self.teacher).decoder.enc_dense_logits
        return t_logits.float(), valid, out["defe"]["density_map"].float()

    def losses(self, student, samples, outputs):
        t_logits, valid, t_density = self._teacher_targets(samples)
        s_logits, _ = _unwrap(student).decoder.enc_dense_logits
        s_logits = s_logits.float()
        assert s_logits.shape == t_logits.shape, (s_logits.shape, t_logits.shape)
        p_t = torch.sigmoid(t_logits)
        bce = F.binary_cross_entropy_with_logits(s_logits, p_t, reduction="none").sum(-1)
        ent = F.binary_cross_entropy_with_logits(t_logits, p_t, reduction="none").sum(-1)
        v = valid[0, :, 0].bool()  # anchor validity is the same for every image ([1, L, 1])
        kl = (bce - ent)[:, v].mean()
        s_density = outputs["defe"]["density_map"].float()
        mse = F.mse_loss(s_density, t_density)
        return {"loss_distill_enc": self.w_enc * kl, "loss_distill_density": self.w_density * mse}
