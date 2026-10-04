"""Learning-rate schedule: linear warm-up, then cosine decay (as in the RIFE paper)."""
import math


def lr_at(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
