"""Checkpoints hold everything needed to resume training exactly where it stopped."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, step: int, best_psnr: float, config: dict, scaler=None) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_psnr": best_psnr,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a crash mid-save never leaves a truncated checkpoint behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, restore_rng: bool = True) -> dict:
    # weights_only=False: our own files also hold Python/NumPy RNG state.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise RuntimeError(
            f"incompatible checkpoint {path}: the model architecture differs "
            f"(check the distill/refine flags in your config)\n{e}"
        ) from e
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        rng = ckpt["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return {"step": ckpt["step"], "best_psnr": ckpt["best_psnr"], "config": ckpt["config"]}
