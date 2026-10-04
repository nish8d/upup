import torch

from rife.loss import compute_losses, lap_loss


def _fake_out(student_merged, teacher_merged, student_flow, teacher_flow):
    return {
        "pred": student_merged,
        "merged": [student_merged],
        "flows": [student_flow],
        "teacher": {"flow": teacher_flow, "mask": None, "merged": teacher_merged},
    }


def test_lap_loss_zero_for_identical_and_positive_otherwise():
    x = torch.rand(2, 3, 32, 32)
    assert lap_loss(x, x).item() == 0.0
    assert lap_loss(x, torch.roll(x, 2, dims=-1)).item() > 0.0


def test_without_teacher_total_is_reconstruction():
    gt = torch.rand(1, 3, 32, 32)
    out = {"pred": torch.rand(1, 3, 32, 32), "merged": [], "flows": [], "teacher": None}
    losses = compute_losses(out, gt)
    assert torch.equal(losses["total"], losses["rec"])
    assert losses["tea"].item() == 0.0 and losses["dis"].item() == 0.0


def test_distillation_only_where_teacher_is_better():
    gt = torch.zeros(1, 3, 32, 32)
    student = torch.zeros(1, 3, 32, 32)
    student[..., :16] = 1.0  # student is wrong on the left half only
    teacher = torch.zeros(1, 3, 32, 32)  # teacher is right everywhere
    out = _fake_out(student, teacher, torch.zeros(1, 4, 32, 32), torch.ones(1, 4, 32, 32))
    # Per-pixel flow L1 = 1, counted on the left half only → mean 0.5.
    assert abs(compute_losses(out, gt)["dis"].item() - 0.5) < 1e-6


def test_no_distillation_when_teacher_is_worse():
    gt = torch.zeros(1, 3, 32, 32)
    out = _fake_out(torch.zeros_like(gt), torch.ones_like(gt), torch.zeros(1, 4, 32, 32), torch.ones(1, 4, 32, 32))
    assert compute_losses(out, gt)["dis"].item() == 0.0


def test_distillation_does_not_update_teacher_flow():
    gt = torch.zeros(1, 3, 32, 32)
    student_flow = torch.zeros(1, 4, 32, 32, requires_grad=True)
    teacher_flow = torch.ones(1, 4, 32, 32, requires_grad=True)
    out = _fake_out(torch.ones_like(gt), torch.zeros_like(gt), student_flow, teacher_flow)
    compute_losses(out, gt)["dis"].backward()
    assert student_flow.grad is not None
    assert teacher_flow.grad is None
