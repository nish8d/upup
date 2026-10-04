import cv2
import numpy as np
import pytest

from rife.config import TrainConfig

SEQUENCES = ["00001/0001", "00001/0002", "00002/0001", "00002/0002"]


@pytest.fixture
def fake_vimeo(tmp_path):
    """A tiny Vimeo-90K-shaped dataset: 4 triplets of 64×96 PNGs with horizontal motion."""
    root = tmp_path / "vimeo_triplet"
    rng = np.random.default_rng(0)
    for seq in SEQUENCES:
        seq_dir = root / "sequences" / seq
        seq_dir.mkdir(parents=True)
        base = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        for k in (1, 2, 3):
            cv2.imwrite(str(seq_dir / f"im{k}.png"), np.roll(base, shift=2 * k, axis=1))
    (root / "tri_trainlist.txt").write_text("\n".join(SEQUENCES) + "\n")
    (root / "tri_testlist.txt").write_text("\n".join(SEQUENCES[:2]) + "\n")
    return root


@pytest.fixture
def make_cfg(fake_vimeo, tmp_path):
    """Tiny CPU-friendly training config over fake_vimeo; pass overrides as kwargs."""

    def _make(**overrides):
        values = dict(
            name="test", data_root=str(fake_vimeo), out_dir=str(tmp_path / "runs"),
            batch_size=2, crop=64, num_workers=0, pin_memory=False, amp=False,
            total_steps=4, warmup_steps=2, val_every=1000, val_subset=2, ckpt_every=1000, log_every=1,
        )
        values.update(overrides)
        return TrainConfig(**values)

    return _make
