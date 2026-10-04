import torch

from rife.metrics import psnr, ssim


def test_psnr_known_value():
    # MSE = 0.01 → PSNR = 10·log10(1/0.01) = 20 dB
    value = psnr(torch.full((1, 3, 16, 16), 0.1), torch.zeros(1, 3, 16, 16))
    assert torch.allclose(value, torch.tensor([20.0]), atol=1e-4)


def test_psnr_identical_is_capped():
    x = torch.rand(2, 3, 16, 16)
    assert torch.allclose(psnr(x, x), torch.tensor([100.0, 100.0]))


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5)


def test_ssim_drops_with_noise():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert ssim(noisy, x).item() < 0.9
