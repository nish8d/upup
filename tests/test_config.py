import math

import pytest

from rife.config import TrainConfig
from rife.schedule import lr_at


def test_yaml_loading_coerces_types(tmp_path):
    path = tmp_path / "c.yaml"
    # PyYAML reads "3e-4" (no dot) as a *string*; the loader must coerce it.
    path.write_text("name: run\ndata_root: data/v\nlr_max: 3e-4\nbatch_size: 8\nrefine: true\n")
    cfg = TrainConfig.from_yaml(path)
    assert cfg.lr_max == pytest.approx(3e-4) and isinstance(cfg.lr_max, float)
    assert cfg.batch_size == 8 and cfg.refine is True
    assert str(cfg.run_dir) == "runs/run"
    assert cfg.to_dict()["batch_size"] == 8


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: run\ndata_root: d\nbatchsize: 8\n")
    with pytest.raises(ValueError, match="batchsize"):
        TrainConfig.from_yaml(path)


def test_missing_required_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: run\n")
    with pytest.raises(ValueError, match="data_root"):
        TrainConfig.from_yaml(path)


def test_bool_must_be_bool(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: run\ndata_root: d\nrefine: 'no'\n")
    with pytest.raises(ValueError, match="refine"):
        TrainConfig.from_yaml(path)


def test_lr_schedule_shape():
    kw = dict(warmup=10, total=110, lr_max=1e-3, lr_min=1e-4)
    assert lr_at(0, **kw) == pytest.approx(1e-4)  # 1/10 of the way through warm-up
    assert lr_at(9, **kw) == pytest.approx(1e-3)
    assert lr_at(10, **kw) == pytest.approx(1e-3)
    assert lr_at(60, **kw) == pytest.approx(1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(110, **kw) == pytest.approx(1e-4)
    after_warmup = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(after_warmup, after_warmup[1:]))


@pytest.mark.parametrize("bad", [
    dict(accum_steps=0), dict(batch_size=0), dict(val_every=0), dict(ckpt_every=0), dict(log_every=0),
    dict(val_subset=0), dict(total_steps=0), dict(warmup_steps=-1),
    dict(total_steps=100, warmup_steps=100), dict(lr_min=1e-2, lr_max=1e-3),
])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        TrainConfig(name="run", data_root="d", **bad)
