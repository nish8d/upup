"""Evaluate a checkpoint on the Vimeo-90K test split and measure 720p inference speed.

Usage:
    python evaluate.py --ckpt runs/final/best.pt
Prints a markdown row for the README results table: | run | PSNR | SSIM | 720p fps |
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rife.data import VimeoTriplet
from rife.evaluation import evaluate_model
from rife.inference import interpolate_pair, load_model_for_inference


def measure_fps(model, device, height: int = 720, width: int = 1280, iters: int = 50, warmup: int = 10) -> float:
    img0 = torch.rand(1, 3, height, width, device=device)
    img1 = torch.rand(1, 3, height, width, device=device)
    for _ in range(warmup):  # cuDNN autotuning and allocator warm-up
        interpolate_pair(model, img0, img1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        interpolate_pair(model, img0, img1)
    torch.cuda.synchronize()  # kernels are async; time only finished work
    return iters / (time.perf_counter() - start)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate a RIFE checkpoint on Vimeo-90K.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="data/vimeo_triplet")
    parser.add_argument("--subset", type=int, help="only the first N test triplets")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-speed", action="store_true", help="skip the 720p fps benchmark")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.ckpt, device)
    test_set = VimeoTriplet(args.data_root, "test", subset=args.subset)
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    metrics = evaluate_model(model, loader, device, amp=device.type == "cuda")
    metrics["fps"] = None if args.no_speed or device.type != "cuda" else measure_fps(model, device)

    fps = "n/a" if metrics["fps"] is None else f"{metrics['fps']:.1f}"
    print(f"evaluated {metrics['n']} triplets")
    print(f"| {Path(args.ckpt).parent.name} | {metrics['psnr']:.2f} | {metrics['ssim']:.4f} | {fps} |")
    return metrics


if __name__ == "__main__":
    main()
