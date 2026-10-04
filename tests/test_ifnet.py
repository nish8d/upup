import torch

from rife.ifnet import IFNet


def test_student_outputs():
    net = IFNet(distill=True)
    img0, img1 = torch.rand(2, 3, 64, 96), torch.rand(2, 3, 64, 96)
    out = net(img0, img1)
    assert len(out["flows"]) == len(out["masks"]) == len(out["merged"]) == 3
    for flow, mask, merged in zip(out["flows"], out["masks"], out["merged"]):
        assert flow.shape == (2, 4, 64, 96)
        assert mask.shape == (2, 1, 64, 96)
        assert merged.shape == (2, 3, 64, 96)
        assert ((mask > 0) & (mask < 1)).all()
    assert out["warped"][0].shape == (2, 3, 64, 96)
    assert out["teacher"] is None  # no gt → no teacher


def test_teacher_runs_only_with_gt_and_distill():
    img0, img1, gt = (torch.rand(1, 3, 64, 64) for _ in range(3))
    out = IFNet(distill=True)(img0, img1, gt)
    assert set(out["teacher"]) == {"flow", "mask", "merged"}
    assert out["teacher"]["flow"].shape == (1, 4, 64, 64)
    assert IFNet(distill=False)(img0, img1, gt)["teacher"] is None


def test_no_teacher_parameters_without_distill():
    assert not any(n.startswith("teacher") for n, _ in IFNet(distill=False).named_parameters())


def test_half_resolution_flow():
    out = IFNet(distill=False)(torch.rand(1, 3, 64, 128), torch.rand(1, 3, 64, 128), scale_factor=0.5)
    assert out["merged"][-1].shape == (1, 3, 64, 128)
