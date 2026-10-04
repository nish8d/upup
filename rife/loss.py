"""Training losses (RIFE paper, Sec. 3.4)."""
import torch
import torch.nn.functional as F


def _gaussian_kernel(channels: int, device) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    k = torch.outer(k, k) / 256.0
    return k.expand(channels, 1, 5, 5).contiguous()


def _laplacian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    kernel = _gaussian_kernel(x.shape[1], x.device)
    pyramid, current = [], x
    for _ in range(levels):
        blurred = F.conv2d(F.pad(current, (2, 2, 2, 2), mode="replicate"), kernel, groups=x.shape[1])
        down = blurred[..., ::2, ::2]
        up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
        pyramid.append(current - up)  # band-pass detail at this scale
        current = down
    pyramid.append(current)  # low-pass residual
    return pyramid


def lap_loss(pred: torch.Tensor, gt: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """L1 over a Laplacian pyramid: penalizes errors at every frequency band, so blur is
    punished more than with a plain per-pixel L1."""
    pred, gt = pred.float(), gt.float()
    return sum(F.l1_loss(a, b) for a, b in zip(_laplacian_pyramid(pred, levels), _laplacian_pyramid(gt, levels)))


def compute_losses(out: dict, gt: torch.Tensor, distill_weight: float = 0.01) -> dict[str, torch.Tensor]:
    gt = gt.float()
    rec = lap_loss(out["pred"], gt)
    zero = rec.new_zeros(())
    teacher = out.get("teacher")
    if teacher is None:
        return {"total": rec, "rec": rec, "tea": zero, "dis": zero}

    tea = lap_loss(teacher["merged"], gt)
    with torch.no_grad():
        # Only learn from the teacher where it actually reconstructs better (small margin
        # as in the official code), so a wrong teacher cannot drag the student along.
        student_err = (out["merged"][-1].float() - gt).abs().mean(1, keepdim=True)
        teacher_err = (teacher["merged"].float() - gt).abs().mean(1, keepdim=True)
        teacher_better = (student_err > teacher_err + 0.01).float()
    target = teacher["flow"].detach().float()
    dis = torch.stack(
        [((flow.float() - target).abs().mean(1, keepdim=True) * teacher_better).mean() for flow in out["flows"]]
    ).mean()
    return {"total": rec + tea + distill_weight * dis, "rec": rec, "tea": tea, "dis": dis}
