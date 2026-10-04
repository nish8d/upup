"""IFNet: coarse-to-fine intermediate flow estimation (RIFE paper, Sec. 3.1)."""
import torch
import torch.nn as nn

from rife.ifblock import IFBlock
from rife.warp import warp


def _blend(img0, img1, flow, mask_logit):
    w0 = warp(img0, flow[:, :2])
    w1 = warp(img1, flow[:, 2:4])
    mask = torch.sigmoid(mask_logit)
    # The mask decides, per pixel, which warped input to trust (e.g. under occlusion).
    return w0, w1, mask, mask * w0 + (1 - mask) * w1


class IFNet(nn.Module):
    scales = (4, 2, 1)

    def __init__(self, distill: bool = True):
        super().__init__()
        # Block 0 sees only the two frames; later blocks also see the current warps, mask and flow.
        self.blocks = nn.ModuleList([IFBlock(6, 240), IFBlock(13 + 4, 150), IFBlock(13 + 4, 90)])
        # Privileged teacher (Sec. 3.3): also sees the ground-truth middle frame. Training only.
        self.teacher = IFBlock(16 + 4, 90) if distill else None

    def forward(self, img0, img1, gt=None, scale_factor: float = 1.0) -> dict:
        flows, masks, merged = [], [], []
        flow = mask_logit = None
        w0, w1 = img0, img1
        for block, s in zip(self.blocks, self.scales):
            scale = s / scale_factor
            if flow is None:
                flow, mask_logit = block(torch.cat([img0, img1], 1), None, scale)
            else:
                d_flow, d_mask = block(torch.cat([img0, img1, w0, w1, mask_logit], 1), flow, scale)
                flow, mask_logit = flow + d_flow, mask_logit + d_mask
            w0, w1, mask, blended = _blend(img0, img1, flow, mask_logit)
            flows.append(flow)
            masks.append(mask)
            merged.append(blended)

        teacher = None
        if self.teacher is not None and gt is not None:
            x = torch.cat([img0, img1, w0, w1, mask_logit, gt], 1)
            d_flow, d_mask = self.teacher(x, flow, 1 / scale_factor)
            t_flow, t_logit = flow + d_flow, mask_logit + d_mask
            _, _, t_mask, t_merged = _blend(img0, img1, t_flow, t_logit)
            teacher = {"flow": t_flow, "mask": t_mask, "merged": t_merged}

        return {"flows": flows, "masks": masks, "merged": merged, "warped": (w0, w1), "teacher": teacher}
