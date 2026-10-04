import torch

from rife.warp import warp


def test_zero_flow_is_identity():
    img = torch.rand(2, 3, 16, 24)
    out = warp(img, torch.zeros(2, 2, 16, 24))
    assert torch.allclose(out, img, atol=1e-5)


def test_unit_x_flow_shifts_left_by_one_pixel():
    # Backward warping: out(x) = img(x + 1).
    img = torch.arange(24, dtype=torch.float32).view(1, 1, 1, 24).expand(1, 1, 8, 24).contiguous()
    flow = torch.zeros(1, 2, 8, 24)
    flow[:, 0] = 1.0
    out = warp(img, flow)
    assert torch.allclose(out[..., :-1], img[..., 1:], atol=1e-4)
    assert torch.allclose(out[..., -1], img[..., -1], atol=1e-4)  # border padding


def test_unit_y_flow_shifts_up_by_one_pixel():
    img = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 24).contiguous()
    flow = torch.zeros(1, 2, 8, 24)
    flow[:, 1] = 1.0
    out = warp(img, flow)
    assert torch.allclose(out[:, :, :-1], img[:, :, 1:], atol=1e-4)


def test_output_is_fp32_even_for_half_inputs():
    img = torch.rand(1, 3, 8, 8).half()
    out = warp(img, torch.zeros(1, 2, 8, 8).half())
    assert out.dtype == torch.float32


def test_gradients_flow_to_image_and_flow():
    img = torch.rand(1, 3, 8, 8, requires_grad=True)
    flow = (torch.rand(1, 2, 8, 8) * 2 - 1).requires_grad_()
    warp(img, flow).sum().backward()
    assert img.grad is not None and flow.grad is not None
