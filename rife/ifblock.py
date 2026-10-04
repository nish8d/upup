"""IFBlock: one coarse-to-fine step of IFNet (RIFE paper, Sec. 3.1, Fig. 3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """3×3 conv + PReLU, the basic unit used throughout RIFE."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1), nn.PReLU(out_ch))


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class IFBlock(nn.Module):
    """Estimate (a correction to) the intermediate flows and fusion mask at 1/scale resolution.

    Working at low resolution gives a large receptive field cheaply, so coarse blocks can
    capture big motions; later blocks at finer scales only need to fix small residuals.
    """

    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.down = nn.Sequential(conv(in_ch, width // 2, stride=2), conv(width // 2, width, stride=2))
        self.body = nn.Sequential(*[conv(width, width) for _ in range(8)])
        self.up = nn.ConvTranspose2d(width, 5, 4, 2, 1)  # 4 flow channels + 1 mask logit

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None, scale: float):
        h, w = x.shape[-2:]
        small = (int(h / scale), int(w / scale))
        x = resize(x, small)
        if flow is not None:
            # Flow vectors are in pixels, so they shrink with the image.
            x = torch.cat([x, resize(flow, small) / scale], dim=1)
        feat = self.down(x)  # 1/(4·scale)
        feat = self.body(feat) + feat
        out = self.up(feat)  # 1/(2·scale)
        # Cast to fp32: flows are summed across blocks and bf16 would lose sub-pixel precision.
        out = resize(out, (h, w)).float()
        return out[:, :4] * (2 * scale), out[:, 4:5]
