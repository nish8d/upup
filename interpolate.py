"""Raise a video's frame rate 2×, 4× or 8× with a trained RIFE checkpoint.

Usage:
    python interpolate.py input.mp4 --ckpt runs/final/best.pt --factor 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from rife.inference import interpolate_recursive, load_model_for_inference
from rife.video import FrameWriter, probe, read_frames, require_ffmpeg

DEPTH = {2: 1, 4: 2, 8: 3}


def is_scene_cut(f0: np.ndarray, f1: np.ndarray, threshold: float) -> bool:
    """Interpolating across a hard cut produces a ghostly double exposure; detect it instead."""
    return float(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).mean()) / 255.0 > threshold


def _to_tensor(frames: list[np.ndarray], device) -> torch.Tensor:
    return torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float() / 255.0


def _to_frames(t: torch.Tensor) -> list[np.ndarray]:
    return list((t * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frame-rate up-conversion with RIFE.")
    p.add_argument("input", type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--factor", type=int, choices=sorted(DEPTH), default=2)
    p.add_argument("--out", type=Path, help="default: <input stem>_<factor>x<suffix>")
    p.add_argument("--scale", type=float, choices=[1.0, 0.5, 0.25], default=1.0,
                   help="flow resolution; use 0.5 for 1080p+ to save memory and time")
    p.add_argument("--batch", type=int, default=4, help="frame pairs per GPU batch")
    p.add_argument("--scene-threshold", type=float, default=0.2,
                   help="mean abs difference (0-1) above which a pair is treated as a scene cut")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    require_ffmpeg()
    info = probe(args.input)
    if info.variable_frame_rate:
        print("warning: variable frame rate input; output timing will be approximate", file=sys.stderr)
    device = torch.device(args.device)
    model = load_model_for_inference(args.ckpt, device)
    out_path = args.out or args.input.with_name(f"{args.input.stem}_{args.factor}x{args.input.suffix}")
    depth = DEPTH[args.factor]

    frames = read_frames(args.input, info)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"{args.input} contains no decodable frames")
    writer = FrameWriter(out_path, info.width, info.height, info.fps * args.factor,
                         audio_source=args.input if info.has_audio else None)

    def flush(pairs) -> int:
        mids = interpolate_recursive(model, _to_tensor([a for a, _ in pairs], device),
                                     _to_tensor([b for _, b in pairs], device), depth, args.scale)
        mids = [_to_frames(m) for m in mids]  # depth-ordered list of per-pair frames
        written = 0
        for i, (f0, f1) in enumerate(pairs):
            cut = is_scene_cut(f0, f1, args.scene_threshold)
            for level in mids:
                writer.write(f0 if cut else level[i])
            writer.write(f1)
            written += len(mids) + 1
        return written

    try:
        writer.write(first)
        count, pairs, prev = 1, [], first
        for frame in frames:
            pairs.append((prev, frame))
            prev = frame
            if len(pairs) == args.batch:
                count += flush(pairs)
                pairs = []
        if pairs:
            count += flush(pairs)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    if count == 1:
        print("warning: input has a single frame; output is a copy", file=sys.stderr)
    print(f"wrote {count} frames at {float(info.fps * args.factor):.3f} fps to {out_path}")
    return count


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
