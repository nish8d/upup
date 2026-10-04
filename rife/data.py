"""Vimeo-90K triplet loading and augmentation.

Augmentation randomness is derived from each sample's *global position* in training (via
InfiniteSampler), not from worker RNG state. That makes a resumed run see exactly the same
crops and flips as an uninterrupted one, regardless of worker count.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

cv2.setNumThreads(0)  # DataLoader workers already parallelize; avoid CPU oversubscription


def augment_triplet(frames: list[np.ndarray], crop: int, rng: random.Random) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    y, x = rng.randint(0, h - crop), rng.randint(0, w - crop)
    frames = [f[y : y + crop, x : x + crop] for f in frames]
    if rng.random() < 0.5:
        frames = [f[:, ::-1] for f in frames]  # horizontal flip
    if rng.random() < 0.5:
        frames = [f[::-1] for f in frames]  # vertical flip
    k = rng.randint(0, 3)
    frames = [np.rot90(f, k) for f in frames]
    if rng.random() < 0.5:
        frames = [f[:, :, ::-1] for f in frames]  # RGB <-> BGR
    if rng.random() < 0.5:
        frames = frames[::-1]  # temporal reversal: the middle frame stays the target
    return [np.ascontiguousarray(f) for f in frames]


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    return img[:, :, ::-1]  # OpenCV loads BGR


def _to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0


class VimeoTriplet(Dataset):
    def __init__(self, root, split: str, crop: int = 256, augment: bool = False, subset: int | None = None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.crop = crop
        self.augment = augment
        list_file = self.root / ("tri_trainlist.txt" if split == "train" else "tri_testlist.txt")
        if not list_file.is_file():
            raise FileNotFoundError(
                f"{list_file} not found — data_root must point at the extracted 'vimeo_triplet' "
                "folder (run scripts/download_vimeo.sh to get it)"
            )
        self.samples = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
        if subset is not None:
            self.samples = self.samples[:subset]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item):
        index, aug_seed = item if isinstance(item, tuple) else (item, None)
        seq_dir = self.root / "sequences" / self.samples[index]
        frames = [_read_rgb(seq_dir / f"im{k}.png") for k in (1, 2, 3)]
        if self.augment:
            frames = augment_triplet(frames, self.crop, random.Random(aug_seed))
        img0, gt, img1 = (_to_tensor(f) for f in frames)
        return img0, gt, img1


class InfiniteSampler(Sampler):
    """Endless shuffled stream of (index, aug_seed), addressable by position for exact resume."""

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self):
        pos = self.start
        while True:
            epoch, offset = divmod(pos, self.n)
            perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            for i in range(offset, self.n):
                yield perm[i], self.seed * 1_000_003 + pos
                pos += 1
