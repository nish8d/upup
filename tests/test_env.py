import pytest
import torch


def test_torch_version_is_recent_enough():
    major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    assert (major, minor) >= (2, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_build_supports_this_gpu():
    # A wheel built without sm_120 imports fine but fails on the first kernel launch.
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(x @ x).all()
