import pytest
import torch

from rife.checkpoint import load_checkpoint, save_checkpoint
from rife.model import RIFE


def _trained_pair():
    torch.manual_seed(0)
    model = RIFE(distill=True, refine=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    out["pred"].mean().backward()
    opt.step()
    return model, opt


def test_roundtrip_restores_model_optimizer_and_metadata(tmp_path):
    model, opt = _trained_pair()
    save_checkpoint(tmp_path / "last.pt", model, opt, step=7, best_psnr=31.5, config={"refine": False})
    assert not list(tmp_path.glob("*.tmp"))

    model2 = RIFE(distill=True, refine=False)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(tmp_path / "last.pt", model2, opt2)
    assert meta == {"step": 7, "best_psnr": 31.5, "config": {"refine": False}}
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)


def test_rng_state_is_restored(tmp_path):
    model, opt = _trained_pair()
    save_checkpoint(tmp_path / "c.pt", model, opt, 0, 0.0, {})
    expected = torch.rand(3)
    load_checkpoint(tmp_path / "c.pt", model, opt)
    assert torch.equal(torch.rand(3), expected)


def test_architecture_mismatch_is_reported(tmp_path):
    save_checkpoint(tmp_path / "c.pt", RIFE(refine=False), None, 0, 0.0, {})
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        load_checkpoint(tmp_path / "c.pt", RIFE(refine=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_disabled_scaler_checkpoint_loads_into_enabled_scaler(tmp_path):
    model, opt = _trained_pair()
    save_checkpoint(tmp_path / "c.pt", model, opt, 1, 0.0, {}, torch.amp.GradScaler("cuda", enabled=False))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    load_checkpoint(tmp_path / "c.pt", RIFE(distill=True, refine=False), scaler=scaler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_checkpoint_from_machine_with_more_gpus_loads(tmp_path):
    model, opt = _trained_pair()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 1, 0.0, {})
    state = torch.load(path, weights_only=False)
    state["rng"]["cuda"] = state["rng"]["cuda"] + [state["rng"]["cuda"][0]]
    assert len(state["rng"]["cuda"]) == torch.cuda.device_count() + 1
    torch.save(state, path)
    load_checkpoint(path, RIFE(distill=True, refine=False))
