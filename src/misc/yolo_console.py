"""Ultralytics-style console output, results.csv and results.png for Dome-DETR training (`console_style: yolo`).

Training prints one header per epoch and progress rows like

      Epoch    GPU_mem       loss   vfl_loss  bbox_loss  giou_loss   fgl_loss  Instances       Size
     13/160      41.2G      17.43     0.5120    0.03410     0.5980     1.2010        215        800  34%|###4      | 600/1752 [07:45<14:52, 1.29it/s]

(losses are running means over the epoch; Instances = objects in the batch; Size = training image size). On a terminal
the row is redrawn in place; in a log file a new row is written every `every_sec` seconds and at the end of the epoch.
Evaluation prints a progress row and a compact table (AP, AP50, AP75, AP_vt, AP_t, AP_s, AP_m) instead of the 12-line
COCO summary. After every epoch a row is appended to <output_dir>/results.csv and <output_dir>/results.png is redrawn.
The JSON log (log.txt) is unchanged."""
import csv
import math
import os
import sys
import time

import torch

LOSS_COLS = [("loss", "loss"), ("loss_vfl", "vfl_loss"), ("loss_bbox", "bbox_loss"), ("loss_giou", "giou_loss"), ("loss_fgl", "fgl_loss")]
EXTRA_COLS = [("loss_distill_enc", "dist_enc"), ("loss_distill_density", "dist_den")]
METRIC_NAMES = ["AP", "AP50", "AP75", "AP_vt", "AP_t", "AP_s", "AP_m"]
METRIC_IDX = [0, 2, 3, 4, 5, 6, 7]  # positions in the AI-TOD evaluator's stats vector (1 = AP25, not computed)


def _bar(frac, width=12):
    full = int(round(frac * width))
    return "\u2501" * full + "\u2500" * (width - full)


def _hms(s):
    s = int(max(s, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


class YoloConsole:
    def __init__(self, epochs, output_dir=None, every_sec=30.0, stream=None):
        self.epochs = epochs
        self.output_dir = str(output_dir) if output_dir else None
        self.every_sec = every_sec
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def _emit(self, text, final):
        if self.tty:
            self.stream.write("\r" + text + ("\n" if final else ""))
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    # ---- training -------------------------------------------------------------------------------------------
    def train_iter(self, data_loader, metric_logger, epoch):
        n = len(data_loader)
        cols = None  # header printed with the first row, once the loss meters exist
        t0 = last = time.time()
        inst, size = 0, 0
        for i, obj in enumerate(data_loader):
            samples, targets = obj
            inst = int(sum(len(t["labels"]) for t in targets))
            size = int(samples.shape[-1])
            yield obj
            now_cols = LOSS_COLS + [(k, h) for k, h in EXTRA_COLS if k in metric_logger.meters]
            if now_cols != cols:
                cols = now_cols
                head = ("%11s" * 2) % ("Epoch", "GPU_mem") + "".join("%11s" % h for _, h in cols) + ("%11s" * 2) % ("Instances", "Size")
                self.stream.write(("\n" if i == 0 else "") + head + "\n")
            now = time.time()
            final = i == n - 1
            if not (final or self.tty or now - last >= self.every_sec or i == 0):
                continue
            last = now
            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            vals = []
            for k, _ in cols:
                m = metric_logger.meters.get(k)
                v = m.global_avg if m is not None and m.count else float("nan")
                vals.append(f"{v:11.4f}" if abs(v) < 10 else f"{v:11.3f}")
            done = i + 1
            rate = done / max(now - t0, 1e-9)
            row = ("%11s" % f"{epoch + 1}/{self.epochs}") + ("%11s" % f"{mem:.1f}G") + "".join(vals) + ("%11d%11d" % (inst, size))
            row += f"  {100 * done // n:3d}%|{_bar(done / n)}| {done}/{n} [{_hms(now - t0)}<{_hms((n - done) / rate)}, {rate:.2f}it/s]"
            self._emit(row, final)

    # ---- evaluation -----------------------------------------------------------------------------------------
    def eval_iter(self, data_loader):
        n = len(data_loader)
        t0 = last = time.time()
        for i, obj in enumerate(data_loader):
            yield obj
            now = time.time()
            final = i == n - 1
            if final or self.tty or now - last >= self.every_sec:
                last = now
                done = i + 1
                rate = done / max(now - t0, 1e-9)
                self._emit(("%22s" % "Validating") + f"  {100 * done // n:3d}%|{_bar(done / n)}| {done}/{n} "
                           f"[{_hms(now - t0)}<{_hms((n - done) / rate)}, {rate:.2f}it/s]", final)

    def eval_table(self, stats, n_images, n_instances, extra=""):
        head = ("%11s" * 2) % ("Images", "Instances") + "".join("%9s" % m for m in METRIC_NAMES)
        vals = ("%11d%11d" % (n_images, n_instances)) + "".join("%9.1f" % (100 * stats[j]) if stats[j] >= 0 else "%9s" % "-" for j in METRIC_IDX)
        self.stream.write(head + "\n" + vals + ("   " + extra if extra else "") + "\n")
        self.stream.flush()

    # ---- results.csv / results.png --------------------------------------------------------------------------
    def log_epoch(self, epoch, train_stats, test_stats=None):
        if not self.output_dir:
            return
        try:
            path = os.path.join(self.output_dir, "results.csv")
            keys = [k for k, _ in LOSS_COLS + EXTRA_COLS] + ["lr"]
            header = ["epoch"] + [f"train/{k}" for k in keys] + [f"metrics/{m}" for m in METRIC_NAMES]
            row = [epoch + 1] + [train_stats.get(k, "") for k in keys]
            s = (test_stats or {}).get("coco_eval_bbox")
            row += [round(100 * s[j], 3) if s and s[j] >= 0 else "" for j in METRIC_IDX]
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(header)
                w.writerow(row)
            self.plot(path)
        except Exception as e:  # cosmetics must never stop training
            print(f"[yolo_console] results.csv/png skipped: {e}", flush=True)

    @staticmethod
    def plot(csv_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        cols = [c for c in rows[0] if c != "epoch" and any(r[c] not in ("", None) for r in rows)]
        nc = 5
        nr = math.ceil(len(cols) / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.8 * nr), constrained_layout=True)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, c in zip(axes, cols):
            pts = [(float(r["epoch"]), float(r[c])) for r in rows if r[c] not in ("", None)]
            x, y = zip(*pts)
            ax.plot(x, y, marker="o" if c.startswith("metrics/") else "", ms=3, lw=1.5, color="#1f77b4")
            ax.set_title(c, fontsize=10)
            ax.grid(alpha=0.25)
        for ax in axes[len(cols):]:
            ax.axis("off")
        fig.savefig(os.path.join(os.path.dirname(csv_path), "results.png"), dpi=110)
        plt.close(fig)
