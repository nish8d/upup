"""Train RIFE on Vimeo-90K triplets.

Usage:
    python train.py --config configs/local.yaml
    python train.py --config configs/local.yaml --resume runs/final/last.pt
    python train.py --config configs/ablation_2_distill.yaml --max-steps 300 --name pilot
"""
import argparse
import random
import signal
import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rife.checkpoint import load_checkpoint, save_checkpoint
from rife.config import TrainConfig
from rife.data import InfiniteSampler, VimeoTriplet
from rife.evaluation import evaluate_model
from rife.loss import compute_losses
from rife.model import RIFE
from rife.schedule import lr_at


def _ignore_sigint(_worker_id: int) -> None:
    # Ctrl-C reaches every process in the terminal's group. Workers must ignore it, or they
    # die and the main process crashes before it can save a checkpoint.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train(
    cfg: TrainConfig,
    resume: str | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    on_step_end: Callable[[int], None] | None = None,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    use_amp = cfg.amp and device.type == "cuda"
    # bf16 needs no loss scaling; GPUs without bf16 (Kaggle T4/P100) fall back to fp16 + GradScaler.
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    model = RIFE(distill=cfg.distill, refine=cfg.refine).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    step, best_psnr = 0, float("-inf")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scaler)
        step, best_psnr = state["step"], state["best_psnr"]
        print(f"resumed from {resume} at step {step}")

    samples_per_step = cfg.batch_size * cfg.accum_steps
    train_set = VimeoTriplet(cfg.data_root, "train", crop=cfg.crop, augment=True)
    val_set = VimeoTriplet(cfg.data_root, "test", subset=cfg.val_subset)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=InfiniteSampler(len(train_set), cfg.seed, start=step * samples_per_step),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=_ignore_sigint,
    )
    val_loader = DataLoader(val_set, batch_size=4, num_workers=min(cfg.num_workers, 4), worker_init_fn=_ignore_sigint)

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nCtrl-C received: finishing this step and saving last.pt ...")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    writer = SummaryWriter(str(run_dir / "tb"))
    end_step = cfg.total_steps if max_steps is None else min(cfg.total_steps, step + max_steps)
    history: list[float] = []
    data_iter = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def checkpoint(name: str) -> None:
        save_checkpoint(run_dir / name, model, optimizer, step, best_psnr, cfg.to_dict(), scaler)

    model.train()
    t_last = time.perf_counter()
    try:
        while step < end_step and not stop_requested:
            lr = lr_at(step, cfg.warmup_steps, cfg.total_steps, cfg.lr_max, cfg.lr_min)
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals: dict[str, float] = {}
            for _ in range(cfg.accum_steps):
                img0, gt, img1 = (t.to(device, non_blocking=True) for t in next(data_iter))
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(img0, img1, gt)
                losses = compute_losses(out, gt, cfg.distill_weight)  # fp32, outside autocast
                scaler.scale(losses["total"] / cfg.accum_steps).backward()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + value.item() / cfg.accum_steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            history.append(totals["total"])

            if step % cfg.log_every == 0:
                now = time.perf_counter()
                samples_per_s = cfg.log_every * samples_per_step / (now - t_last)
                t_last = now
                for key, value in totals.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("perf/samples_per_s", samples_per_s, step)
                mem = ""
                if device.type == "cuda":
                    peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    writer.add_scalar("perf/peak_mem_gb", peak_gb, step)
                    mem = f" peak {peak_gb:.2f} GB"
                print(f"step {step} loss {totals['total']:.4f} lr {lr:.2e} {samples_per_s:.1f} samples/s{mem}")

            if step % cfg.val_every == 0:
                metrics = evaluate_model(model, val_loader, device, amp=use_amp)
                writer.add_scalar("val/psnr", metrics["psnr"], step)
                writer.add_scalar("val/ssim", metrics["ssim"], step)
                print(f"step {step} val PSNR {metrics['psnr']:.2f} dB SSIM {metrics['ssim']:.4f}")
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    checkpoint("best.pt")

            if step % cfg.ckpt_every == 0:
                checkpoint("last.pt")
            if on_step_end is not None:
                on_step_end(step)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        writer.close()

    checkpoint("last.pt")
    return {"step": step, "best_psnr": best_psnr, "history": history}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train RIFE on Vimeo-90K triplets.")
    parser.add_argument("--config", required=True, help="YAML run config")
    parser.add_argument("--resume", help="checkpoint to resume from (e.g. runs/<name>/last.pt)")
    parser.add_argument("--max-steps", type=int, help="stop after this many steps (for pilots)")
    parser.add_argument("--name", help="override the run name from the config")
    args = parser.parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config)
    if args.name:
        cfg.name = args.name
    result = train(cfg, resume=args.resume, max_steps=args.max_steps)
    print(f"stopped at step {result['step']} (best val PSNR {result['best_psnr']:.2f} dB)")


if __name__ == "__main__":
    main()
