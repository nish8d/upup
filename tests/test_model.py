import torch

from rife.model import RIFE


def test_pred_without_refine_is_last_merge():
    model = RIFE(distill=False, refine=False)
    out = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    assert torch.equal(out["pred"], out["merged"][-1])


def test_refine_changes_pred_and_stays_in_range():
    torch.manual_seed(0)
    model = RIFE(distill=False, refine=True)
    out = model(torch.rand(2, 3, 64, 96), torch.rand(2, 3, 64, 96))
    assert out["pred"].shape == (2, 3, 64, 96)
    assert out["pred"].min() >= 0 and out["pred"].max() <= 1
    assert not torch.allclose(out["pred"], out["merged"][-1])


def test_parameter_count_matches_paper_scale():
    # RIFE paper reports ~9.8M parameters for the full model.
    n = sum(p.numel() for p in RIFE(distill=False, refine=True).parameters())
    assert 5e6 < n < 12e6


def test_teacher_keys_are_namespaced():
    keys = RIFE(distill=True).state_dict().keys()
    assert any(k.startswith("ifnet.teacher.") for k in keys)
