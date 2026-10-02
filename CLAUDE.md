# CLAUDE.md

## Project

From-scratch PyTorch reimplementation of RIFE (ECCV 2022) for video frame interpolation:
predict flows `F_t→0`, `F_t→1` from the missing middle frame, backward-warp both inputs,
blend with a learned mask. Trained on Vimeo-90K triplets. Learning/portfolio project that
should also be a practical near-real-time video tool.

Design spec: `docs/superpowers/specs/2026-10-02-rife-interpolation-design.md` — the
source of truth for architecture, losses, training recipe, and CLI behavior.

## Hardware

- Local RTX 5060, 8 GB VRAM (Blackwell, sm_120) — PyTorch must be a CUDA ≥ 12.8 build.
- 16 GB system RAM. Keep VRAM and RAM budgets in mind (batch 16 at 256×256 crops, bf16).
- Optional parallel runs on Kaggle; code must not hardcode paths.

## Layout

- `rife/` — model and training library (`warp`, `ifblock`, `ifnet`, `refine`, `loss`, `data`)
- `train.py`, `evaluate.py`, `interpolate.py` — entry points
- `configs/` — one YAML per run; local vs Kaggle differ only in paths and batch size
- `tests/` — pytest

## Commands

```bash
source .venv/bin/activate
pytest                                           # run tests
python train.py --config configs/local.yaml      # train
python train.py --config configs/local.yaml --resume runs/<name>/last.pt
python evaluate.py --ckpt runs/<name>/best.pt    # PSNR/SSIM + 720p fps
python interpolate.py in.mp4 --factor 2 --ckpt runs/<name>/best.pt
```

## Conventions

- Images are RGB float in [0, 1], shape `B×3×H×W`.
- Warping and losses run in fp32 even under bf16/fp16 autocast (`grid_sample` in low
  precision causes artifacts).
- All run settings live in YAML configs; no hardcoded dataset or output paths.
- Training must stay exactly resumable: any new stateful component (optimizer, scheduler,
  RNG, data sampler) must be saved to and restored from the checkpoint.
- The teacher block is training-only; inference code paths must work with `gt=None`.
- Code is meant to be read and learned from: favor clarity, and add short comments that
  explain *why* (with references to the RIFE paper where relevant).
