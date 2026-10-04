"""The public RIFE model: IFNet (+ training-only teacher) and an optional refinement net."""
import torch.nn as nn

from rife.ifnet import IFNet
from rife.refine import RefineNet


class RIFE(nn.Module):
    def __init__(self, distill: bool = True, refine: bool = False):
        super().__init__()
        self.ifnet = IFNet(distill=distill)
        self.refine = RefineNet() if refine else None

    def forward(self, img0, img1, gt=None, scale_factor: float = 1.0) -> dict:
        out = self.ifnet(img0, img1, gt, scale_factor)
        pred = out["merged"][-1]
        if self.refine is not None:
            warped0, warped1 = out["warped"]
            pred = self.refine(img0, img1, warped0, warped1, out["masks"][-1], out["flows"][-1], pred)
        out["pred"] = pred
        return out
