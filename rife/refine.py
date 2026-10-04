"""Refinement network (RIFE paper, Sec. 3.2): ContextNet features + a small U-Net residual."""
import torch
import torch.nn as nn

from rife.ifblock import conv, resize
from rife.warp import warp


def deconv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.PReLU(out_ch))


class Conv2(nn.Module):
    """Two 3×3 convs; the first halves the resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(conv(in_ch, out_ch, stride=2), conv(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class ContextNet(nn.Module):
    """4-level feature pyramid of one input frame, each level warped toward time t."""

    def __init__(self, c: int = 16):
        super().__init__()
        self.levels = nn.ModuleList([Conv2(3, c), Conv2(c, 2 * c), Conv2(2 * c, 4 * c), Conv2(4 * c, 8 * c)])

    def forward(self, img, flow):
        feats, x = [], img
        for level in self.levels:
            x = level(x)
            flow = resize(flow, x.shape[-2:]) * 0.5  # each level halves the resolution
            feats.append(warp(x, flow))
        return feats


class UNet(nn.Module):
    def __init__(self, c: int = 16):
        super().__init__()
        self.down0 = Conv2(17, 2 * c)  # img0, img1, warped0, warped1 (12) + mask (1) + flow (4)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.out = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped0, warped1, mask, flow, ctx0, ctx1):
        s0 = self.down0(torch.cat([img0, img1, warped0, warped1, mask, flow], 1))
        s1 = self.down1(torch.cat([s0, ctx0[0], ctx1[0]], 1))
        s2 = self.down2(torch.cat([s1, ctx0[1], ctx1[1]], 1))
        s3 = self.down3(torch.cat([s2, ctx0[2], ctx1[2]], 1))
        x = self.up0(torch.cat([s3, ctx0[3], ctx1[3]], 1))
        x = self.up1(torch.cat([x, s2], 1))
        x = self.up2(torch.cat([x, s1], 1))
        x = self.up3(torch.cat([x, s0], 1))
        return self.out(x)


class RefineNet(nn.Module):
    def __init__(self, c: int = 16):
        super().__init__()
        self.context = ContextNet(c)
        self.unet = UNet(c)

    def forward(self, img0, img1, warped0, warped1, mask, flow, merged):
        ctx0 = self.context(img0, flow[:, :2])
        ctx1 = self.context(img1, flow[:, 2:4])
        raw = self.unet(img0, img1, warped0, warped1, mask, flow, ctx0, ctx1).float()
        # A bounded residual: the refiner fixes details, it cannot repaint the frame.
        residual = torch.sigmoid(raw) * 2 - 1
        return (merged + residual).clamp(0, 1)
