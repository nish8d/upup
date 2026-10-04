"""Score a model on a loader of (img0, gt, img1) triplets."""
import torch

from rife.inference import interpolate_pair
from rife.metrics import psnr, ssim


@torch.no_grad()
def evaluate_model(model, loader, device, amp: bool = True) -> dict:
    was_training = model.training
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0
    for img0, gt, img1 in loader:
        img0, gt, img1 = img0.to(device), gt.to(device), img1.to(device)
        pred = interpolate_pair(model, img0, img1, amp=amp)
        pred = torch.round(pred * 255) / 255  # score what would actually be saved as 8-bit
        total_psnr += psnr(pred, gt).sum().item()
        total_ssim += ssim(pred, gt).sum().item()
        n += img0.shape[0]
    model.train(was_training)
    return {"psnr": total_psnr / n, "ssim": total_ssim / n, "n": n}
