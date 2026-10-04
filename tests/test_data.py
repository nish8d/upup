import itertools
import random

import numpy as np
import pytest
import torch

from rife.data import InfiniteSampler, VimeoTriplet, augment_triplet


def test_train_sample_is_cropped(fake_vimeo):
    ds = VimeoTriplet(fake_vimeo, "train", crop=32, augment=True)
    assert len(ds) == 4
    img0, gt, img1 = ds[(0, 123)]
    for t in (img0, gt, img1):
        assert t.shape == (3, 32, 32) and t.dtype == torch.float32
        assert 0.0 <= t.min() and t.max() <= 1.0


def test_same_aug_seed_gives_same_sample(fake_vimeo):
    ds = VimeoTriplet(fake_vimeo, "train", crop=32, augment=True)
    for a, b in zip(ds[(1, 7)], ds[(1, 7)]):
        assert torch.equal(a, b)


def test_test_split_is_full_resolution_and_subsettable(fake_vimeo):
    ds = VimeoTriplet(fake_vimeo, "test")
    assert len(ds) == 2
    assert ds[0][0].shape == (3, 64, 96)
    assert len(VimeoTriplet(fake_vimeo, "test", subset=1)) == 1


@pytest.mark.parametrize("seed", range(20))
def test_augmentation_keeps_triplet_consistent(seed):
    base = np.random.default_rng(seed).integers(0, 250, size=(40, 48, 3), dtype=np.uint8)
    frames = [base, base + 1, base + 2]  # middle frame = base + 1
    out = augment_triplet(frames, crop=32, rng=random.Random(seed))
    mid = out[1].astype(int)
    # Spatial/color ops are identical across frames, and temporal reversal keeps the middle.
    outer = sorted([out[0].astype(int) - mid, out[2].astype(int) - mid], key=lambda d: d.mean())
    assert (outer[0] == -1).all() and (outer[1] == 1).all()
    assert out[1].shape == (32, 32, 3)


def test_missing_root_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="tri_trainlist.txt.*download_vimeo.sh"):
        VimeoTriplet(tmp_path / "nope", "train")


def test_sampler_is_resumable_and_covers_each_epoch():
    full = list(itertools.islice(InfiniteSampler(5, seed=3), 13))
    resumed = list(itertools.islice(InfiniteSampler(5, seed=3, start=7), 6))
    assert resumed == full[7:]
    assert sorted(i for i, _ in full[:5]) == list(range(5))
    assert sorted(i for i, _ in full[5:10]) == list(range(5))
    assert len({s for _, s in full}) == 13  # unique augmentation seed per position
