import math
import os
import signal

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rife.data import VimeoTriplet
from rife.evaluation import evaluate_model
from rife.loss import compute_losses
from rife.model import RIFE
import train as train_module
from train import train


def test_evaluate_model_reports_metrics_and_restores_mode(fake_vimeo):
    torch.manual_seed(0)
    model = RIFE(distill=False).train()
    loader = DataLoader(VimeoTriplet(fake_vimeo, "test"), batch_size=2)
    metrics = evaluate_model(model, loader, torch.device("cpu"), amp=False)
    assert metrics["n"] == 2
    assert math.isfinite(metrics["psnr"]) and -1 <= metrics["ssim"] <= 1
    assert model.training


def test_training_writes_checkpoints_and_validates(make_cfg):
    cfg = make_cfg(val_every=2)
    result = train(cfg, device="cpu")
    assert result["step"] == 4 and len(result["history"]) == 4
    assert (cfg.run_dir / "last.pt").is_file() and (cfg.run_dir / "best.pt").is_file()
    assert math.isfinite(result["best_psnr"])


def test_resume_reproduces_uninterrupted_run(make_cfg):
    full = train(make_cfg(name="full"), device="cpu")
    cfg = make_cfg(name="split")
    train(cfg, max_steps=2, device="cpu")
    resumed = train(cfg, resume=str(cfg.run_dir / "last.pt"), device="cpu")
    assert resumed["step"] == 4
    assert resumed["history"] == pytest.approx(full["history"][2:], rel=1e-5)


def test_ctrl_c_saves_checkpoint_and_stops(make_cfg):
    cfg = make_cfg(total_steps=10)
    handler_before = signal.getsignal(signal.SIGINT)

    def interrupt(step):
        if step == 2:
            os.kill(os.getpid(), signal.SIGINT)

    result = train(cfg, device="cpu", on_step_end=interrupt)
    assert result["step"] == 2
    assert torch.load(cfg.run_dir / "last.pt", weights_only=False)["step"] == 2
    assert signal.getsignal(signal.SIGINT) is handler_before


def _nan_losses(real, bad_calls):
    """Wrap compute_losses so the listed (0-based) calls return a NaN loss that still has a graph."""
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        losses = real(*args, **kwargs)
        calls["n"] += 1
        if bad_calls is None or calls["n"] - 1 in bad_calls:
            losses = {k: v + float("nan") for k, v in losses.items()}
        return losses

    return wrapped


def test_single_nonfinite_step_is_skipped(make_cfg, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "compute_losses", _nan_losses(compute_losses, {1}))
    cfg = make_cfg()
    result = train(cfg, device="cpu")
    assert result["step"] == 4  # the skipped step still counts, so resume data positions stay exact
    assert math.isnan(result["history"][1]) and all(math.isfinite(h) for h in result["history"][2:])
    ckpt = torch.load(cfg.run_dir / "last.pt", weights_only=False)
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point())
    assert "non-finite" in capsys.readouterr().out


def test_persistent_nonfinite_loss_aborts_without_saving(make_cfg, monkeypatch):
    monkeypatch.setattr(train_module, "compute_losses", _nan_losses(compute_losses, None))
    cfg = make_cfg(total_steps=20, warmup_steps=2)
    handler_before = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="non-finite"):
        train(cfg, device="cpu")
    assert not (cfg.run_dir / "last.pt").exists()
    assert signal.getsignal(signal.SIGINT) is handler_before


def test_resume_warns_about_config_drift(make_cfg, capsys):
    cfg = make_cfg(name="drift")
    train(cfg, max_steps=2, device="cpu")
    capsys.readouterr()
    train(make_cfg(name="drift", lr_max=1e-4), resume=str(cfg.run_dir / "last.pt"), device="cpu")
    assert "lr_max" in capsys.readouterr().out


def test_second_ctrl_c_still_interrupts(make_cfg):
    cfg = make_cfg(total_steps=10)
    handler_before = signal.getsignal(signal.SIGINT)

    def interrupt_twice(step):
        if step == 2:
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        train(cfg, device="cpu", on_step_end=interrupt_twice)
    assert signal.getsignal(signal.SIGINT) is handler_before


@pytest.mark.slow
def test_model_overfits_a_single_batch():
    # Sanity check that gradients, warping and losses are wired together correctly:
    # a smooth texture moving 4 px must become easy to interpolate.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    texture = F.interpolate(torch.rand(2, 3, 8, 8), size=(64, 64), mode="bilinear", align_corners=False)
    img0, gt, img1 = (torch.roll(texture, s, dims=-1).to(device) for s in (2, 0, -2))
    model = RIFE(distill=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(150):
        loss = compute_losses(model(img0, img1, gt), gt)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
