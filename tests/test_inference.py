import pytest
import torch

from rife.checkpoint import save_checkpoint
from rife.inference import interpolate_pair, interpolate_recursive, load_model_for_inference, pad_to_multiple
from rife.model import RIFE


def test_pad_to_multiple():
    x = torch.rand(1, 3, 65, 97)
    padded, (h, w) = pad_to_multiple(x, 32)
    assert padded.shape == (1, 3, 96, 128) and (h, w) == (65, 97)
    assert torch.equal(padded[..., :65, :97], x)


@pytest.mark.parametrize("scale_factor", [1.0, 0.5])
def test_interpolate_pair_handles_odd_sizes(scale_factor):
    model = RIFE(distill=False).eval()
    out = interpolate_pair(model, torch.rand(1, 3, 37, 53), torch.rand(1, 3, 37, 53), scale_factor)
    assert out.shape == (1, 3, 37, 53)
    assert out.min() >= 0 and out.max() <= 1


def test_recursive_returns_frames_in_order():
    model = RIFE(distill=False).eval()
    frames = interpolate_recursive(model, torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32), depth=2)
    assert len(frames) == 3
    assert all(f.shape == (1, 3, 32, 32) for f in frames)


def test_load_model_drops_teacher(tmp_path):
    save_checkpoint(tmp_path / "c.pt", RIFE(distill=True, refine=False), None, 0, 0.0, {"refine": False})
    model = load_model_for_inference(tmp_path / "c.pt", "cpu")
    assert model.ifnet.teacher is None and model.refine is None and not model.training


def test_load_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_model_for_inference(tmp_path / "missing.pt", "cpu")
    torch.save({"foo": 1}, tmp_path / "junk.pt")
    with pytest.raises(ValueError, match="not a RIFE checkpoint"):
        load_model_for_inference(tmp_path / "junk.pt", "cpu")


def test_corrupt_checkpoint_is_a_clear_error(tmp_path):
    (tmp_path / "bad.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(ValueError, match="could not read checkpoint"):
        load_model_for_inference(tmp_path / "bad.pt", "cpu")
