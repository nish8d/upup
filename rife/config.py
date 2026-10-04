"""Run configuration. One YAML file per run; unknown keys are errors so typos can't hide."""
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    name: str
    data_root: str
    out_dir: str = "runs"
    distill: bool = True
    refine: bool = False
    batch_size: int = 16
    accum_steps: int = 1
    total_steps: int = 50_000
    warmup_steps: int = 2_000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 1e-4
    distill_weight: float = 0.01
    crop: int = 256
    num_workers: int = 6
    pin_memory: bool = True
    amp: bool = True
    val_every: int = 2_000
    val_subset: int = 500
    ckpt_every: int = 1_000
    log_every: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        at_least_one = ("batch_size", "accum_steps", "total_steps", "val_every", "val_subset",
                        "ckpt_every", "log_every")
        for key in at_least_one:
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be at least 1, got {getattr(self, key)}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) must be less than total_steps ({self.total_steps})")
        if self.lr_min > self.lr_max:
            raise ValueError(f"lr_min ({self.lr_min}) must not exceed lr_max ({self.lr_max})")

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        by_name = {f.name: f for f in fields(cls)}
        unknown = sorted(set(raw) - set(by_name))
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {unknown}")
        missing = sorted(n for n, f in by_name.items() if f.default is MISSING and n not in raw)
        if missing:
            raise ValueError(f"missing required config keys in {path}: {missing}")
        values = {}
        for key, value in raw.items():
            typ = by_name[key].type
            if typ is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"config key {key!r} in {path} must be true/false, got {value!r}")
                values[key] = value
            else:
                values[key] = typ(value)  # e.g. "3e-4" -> 0.0003
        return cls(**values)
