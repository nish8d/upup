import cv2
import numpy as np
import pytest

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
