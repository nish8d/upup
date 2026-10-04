"""Image quality metrics. Inputs are B×C×H×W in [0, 1]; outputs are per-image (shape B)."""
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred.float() - gt.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(channels: int, device, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Standard SSIM (Wang et al. 2004), averaged over channels and valid pixels.
    Values can differ slightly from the RIFE paper's MATLAB-style implementation."""
    pred, gt = pred.float(), gt.float()
    channels = pred.shape[1]
    window = _gaussian_window(channels, pred.device)

    def filt(x):
        return F.conv2d(x, window, groups=channels)

    mu_x, mu_y = filt(pred), filt(gt)
    sigma_xx = filt(pred * pred) - mu_x**2
    sigma_yy = filt(gt * gt) - mu_y**2
    sigma_xy = filt(pred * gt) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_xx + sigma_yy + c2))
    return ssim_map.flatten(1).mean(1)
