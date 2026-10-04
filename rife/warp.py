"""Backward warping: sample a frame along a per-pixel flow field.

RIFE estimates flows *from* the missing frame t *to* the inputs (F_t->0, F_t->1), so every
output pixel knows where to look in the input. That is backward warping, which grid_sample
does directly, without the holes and collisions of forward splatting.
"""
import torch
import torch.nn.functional as F

# Normalized base grids, keyed by (device, H, W). Rebuilding them every call is wasted work.
_grid_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}


def _base_grid(device: torch.device, h: int, w: int) -> torch.Tensor:
    key = (device, h, w)
    if key not in _grid_cache:
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        _grid_cache[key] = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # 1×H×W×2, (x, y) order
    return _grid_cache[key]


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Return img sampled at p + flow(p). flow is B×2×H×W in pixels, channel 0 = x, 1 = y."""
    _, _, h, w = img.shape
    # grid_sample in half precision produces visible artifacts, so always warp in fp32.
    with torch.autocast(device_type=img.device.type, enabled=False):
        img, flow = img.float(), flow.float()
        # With align_corners=True, pixel i maps to -1 + 2i/(size-1), so 1 px = 2/(size-1).
        scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=img.device)
        grid = _base_grid(img.device, h, w) + flow.permute(0, 2, 3, 1) * scale
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
