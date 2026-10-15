"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import datetime
import json
import os
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def _seed_epoch(self, epoch):
        """Reseed every RNG from (run seed, epoch) so an epoch's data order, worker augmentations, multi-scale choices
        and denoising noise depend only on the epoch, never on how the run got there (exact resume)."""
        import random as _random
        import numpy as _np
        base = int(getattr(self.cfg, "seed", None) or 0)
        s = (base * 1_000_003 + int(epoch) * 7_919 + 12_345) % (2**31 - 1)
        _random.seed(s)
        _np.random.seed(s)
        torch.manual_seed(s)  # also seeds every CUDA device
        return s

    def _fit_state(self, best_stat, best_stat_print, top1):
        return {"last_epoch": self.last_epoch, "best_stat": best_stat, "best_stat_print": best_stat_print, "top1": top1,
                "not_improved_count": self.not_improved_count, "stage2_reload": self.stage2_reload,
                "ema_decay": self.ema.decay if self.ema is not None else None}

    def _save_last(self, best_stat, best_stat_print, top1):
        """Write last.pth atomically at the very end of an epoch (after evaluation and every bookkeeping step that can
        change model, EMA or fit state), with the fit state inside, so resuming from it is the same as not stopping."""
        if not self.output_dir or not dist_utils.is_main_process():
            return
        state = self.state_dict()
        state["fit_state"] = self._fit_state(best_stat, best_stat_print, top1)
        tmp = self.output_dir / "last.pth.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.output_dir / "last.pth")

    def _write_fit_state(self, path, best_stat, best_stat_print, top1):
        if path is None:
            return
        fs = {"last_epoch": self.last_epoch, "best_stat": best_stat, "best_stat_print": best_stat_print, "top1": top1,
              "not_improved_count": self.not_improved_count, "stage2_reload": self.stage2_reload,
              "ema_decay": self.ema.decay if self.ema is not None else None}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(fs))
        tmp.replace(path)

    def fit(
        self,
    ):
        self.train()
        args = self.cfg

        if isinstance(self.cfg.train_dataloader.collate_fn.base_size, str):
            try:
                self.cfg.train_dataloader.collate_fn.base_size = int(self.cfg.train_dataloader.collate_fn.base_size)
            except ValueError:
                try:
                    self.cfg.train_dataloader.collate_fn.base_size = self.cfg.train_dataloader.collate_fn.base_size.strip('()[]')
                    self.cfg.train_dataloader.collate_fn.base_size = tuple(int(x.strip()) for x in self.cfg.train_dataloader.collate_fn.base_size.split(','))
                except:
                    raise ValueError(f"Cannot convert base_size string '{self.cfg.train_dataloader.collate_fn.base_size}' to valid size format")

        self.patience = 6
        self.not_improved_count = 0
        self.stage2_reload = "stage1"

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        restored = getattr(self, "_resume_fit_state", None)
        if self.last_epoch > 0 and restored is None:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        fit_state_path = self.output_dir / "fit_state.json" if self.output_dir else None
        fs = restored
        if fs is None and self.last_epoch > 0 and fit_state_path is not None and fit_state_path.exists():
            fs = json.loads(fit_state_path.read_text())
        if fs is not None:
            if fs.get("last_epoch") == self.last_epoch:
                best_stat, best_stat_print, top1 = fs["best_stat"], fs["best_stat_print"], fs["top1"]
                self.not_improved_count, self.stage2_reload = fs["not_improved_count"], fs["stage2_reload"]
                if self.ema is not None:
                    self.ema.decay = fs["ema_decay"]
                print(f"Restored fit state from {fit_state_path}: {fs}")
        eval_interval = int(args.yaml_cfg.get("eval_interval", 1) or 1)
        # "released": stage-2 start and patience reloads of best_stg1 as in the released code; "none": no reloads at all
        reload_policy = str(args.yaml_cfg.get("reload_policy", "released"))
        assert reload_policy in ("released", "none"), reload_policy
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            self._seed_epoch(epoch)
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch and reload_policy == "none":
                # no stage-2 reload: keep training from this epoch's own weights; only the EMA restart decay is applied
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Stage 2 at epoch {epoch} without reload (reload_policy=none), EMA decay {self.ema.decay}")
            elif epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                # the reload also overwrote the epoch counter with the checkpoint's; keep the true one (the reloaded
                # weights/optimizer/scheduler are exactly what the released recipe uses, only the counter is fixed)
                self.last_epoch = epoch - 1
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            print("Train starting...")


            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )


            print("Training state finished.")

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and (epoch + 1) % args.checkpoint_freq == 0:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / f"checkpoint{epoch:04}.pth")

            module = self.ema.module if self.ema else self.model

            if (epoch + 1) % eval_interval != 0 and epoch != args.epoches - 1 and epoch + 1 != self.train_dataloader.collate_fn.stop_epoch:
                if self.output_dir and dist_utils.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps({**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}) + "\n")
                    self._write_fit_state(fit_state_path, best_stat, best_stat_print, top1)
                self._save_last(best_stat, best_stat_print, top1)
                continue

            print("Evaluate state starting...")

            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    self.not_improved_count = 0
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )
                else:
                    self.not_improved_count += 1

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"current_stat: {test_stats[k][0]}")
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch and reload_policy == "none":
                    pass  # no patience reloads / EMA-decay search (reload_policy=none)
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if self.not_improved_count >= self.patience:
                        best_stat = {
                            "epoch": -1,
                        }
                        self.ema.decay -= 0.0001
                        self.not_improved_count = 0
                        if self.stage2_reload == "stage2":
                            self.load_resume_state(str(self.output_dir / "best_stg2.pth"))
                        else:
                            self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        self.last_epoch = epoch  # true epoch (the reload reset the counter to the checkpoint's)
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
                    else:
                        print(f"Tolerate undesirable result for patience: {self.not_improved_count} / {self.patience} ")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                self._write_fit_state(fit_state_path, best_stat, best_stat_print, top1)

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

            # last: everything above (evaluation, best checkpoints, EMA/stage reloads) is now part of the state
            self._save_last(best_stat, best_stat_print, top1)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(
        self,
    ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
