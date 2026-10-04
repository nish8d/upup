"""Inference helpers shared by evaluation and the video CLI."""
from pathlib import Path

import torch
import torch.nn.functional as F

from rife.model import RIFE


def pad_to_multiple(x: torch.Tensor, multiple: int):
    h, w = x.shape[-2:]
    padded = F.pad(x, (0, (-w) % multiple, 0, (-h) % multiple), mode="replicate")
    return padded, (h, w)


@torch.no_grad()
def interpolate_pair(model, img0, img1, scale_factor: float = 1.0, amp: bool = True) -> torch.Tensor:
    # The coarsest block works at 1/(32/scale_factor) resolution, so sizes must divide by that.
    multiple = int(32 / scale_factor)
    p0, (h, w) = pad_to_multiple(img0, multiple)
    p1, _ = pad_to_multiple(img1, multiple)
    with torch.autocast(device_type=img0.device.type, dtype=torch.float16, enabled=amp and img0.is_cuda):
        out = model(p0, p1, scale_factor=scale_factor)
    return out["pred"][..., :h, :w].float().clamp(0, 1)


def interpolate_recursive(model, img0, img1, depth: int, scale_factor: float = 1.0, amp: bool = True):
    """Midpoint recursion: depth 1 → 2×, depth 2 → 4×, depth 3 → 8×."""
    if depth == 0:
        return []
    mid = interpolate_pair(model, img0, img1, scale_factor, amp)
    left = interpolate_recursive(model, img0, mid, depth - 1, scale_factor, amp)
    right = interpolate_recursive(model, mid, img1, depth - 1, scale_factor, amp)
    return left + [mid] + right


def load_model_for_inference(path, device) -> RIFE:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{path} is not a RIFE checkpoint (missing 'model'/'config')")
    model = RIFE(distill=False, refine=bool(ckpt["config"].get("refine", False)))
    # The teacher is training-only, so its weights are simply dropped.
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("ifnet.teacher.")}
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(f"incompatible checkpoint {path}: {e}") from e
    return model.to(device).eval()
