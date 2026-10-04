import pytest
import torch

from rife.ifblock import IFBlock, resize


@pytest.mark.parametrize("scale", [4, 2, 1, 8])
def test_output_shapes_at_each_scale(scale):
    block = IFBlock(in_ch=6, width=32)
    flow, mask = block(torch.rand(2, 6, 64, 96), None, scale)
    assert flow.shape == (2, 4, 64, 96)
    assert mask.shape == (2, 1, 64, 96)


def test_block_with_incoming_flow():
    block = IFBlock(in_ch=13 + 4, width=32)
    flow, mask = block(torch.rand(1, 13, 64, 64), torch.rand(1, 4, 64, 64), 2)
    assert flow.shape == (1, 4, 64, 64)
    assert mask.shape == (1, 1, 64, 64)


def test_outputs_are_fp32_under_autocast():
    block = IFBlock(in_ch=6, width=32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        flow, mask = block(torch.rand(1, 6, 32, 32), None, 1)
    # Flows are accumulated across blocks; bf16 would lose sub-pixel precision.
    assert flow.dtype == torch.float32 and mask.dtype == torch.float32


def test_resize_noop_when_size_matches():
    x = torch.rand(1, 3, 8, 8)
    assert resize(x, (8, 8)) is x
