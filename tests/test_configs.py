from pathlib import Path

import pytest

from rife.config import TrainConfig

CONFIGS = sorted(Path("configs").glob("*.yaml"))


def test_all_expected_configs_exist():
    names = {p.stem for p in CONFIGS}
    assert {"ablation_1_ifnet", "ablation_2_distill", "ablation_3_refine", "local", "kaggle"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    cfg = TrainConfig.from_yaml(path)
    assert cfg.name == path.stem or path.stem in ("local", "kaggle")


def test_ablations_differ_only_in_stage_flags():
    a1, a2, a3 = (TrainConfig.from_yaml(f"configs/{n}.yaml").to_dict()
                  for n in ("ablation_1_ifnet", "ablation_2_distill", "ablation_3_refine"))
    for cfg in (a1, a2, a3):
        cfg.pop("name")
    assert (a1["distill"], a1["refine"]) == (False, False)
    assert (a2["distill"], a2["refine"]) == (True, False)
    assert (a3["distill"], a3["refine"]) == (True, True)
    for cfg in (a1, a2, a3):
        cfg.pop("distill"), cfg.pop("refine")
    assert a1 == a2 == a3  # same step budget and recipe → fair comparison


def test_kaggle_differs_from_local_only_in_environment_keys():
    local = TrainConfig.from_yaml("configs/local.yaml").to_dict()
    kaggle = TrainConfig.from_yaml("configs/kaggle.yaml").to_dict()
    differing = {k for k in local if local[k] != kaggle[k]}
    assert differing <= {"data_root", "out_dir", "batch_size", "num_workers"}
