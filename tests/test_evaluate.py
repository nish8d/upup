import math

from evaluate import main
from rife.checkpoint import save_checkpoint
from rife.model import RIFE


def test_evaluate_prints_table_row(fake_vimeo, tmp_path, capsys):
    ckpt = tmp_path / "myrun" / "best.pt"
    save_checkpoint(ckpt, RIFE(distill=True, refine=False), None, 0, 0.0, {"refine": False})
    metrics = main(["--ckpt", str(ckpt), "--data-root", str(fake_vimeo), "--no-speed", "--num-workers", "0"])
    assert metrics["n"] == 2 and math.isfinite(metrics["psnr"])
    row = capsys.readouterr().out.strip().splitlines()[-1]
    assert row.startswith("| myrun |") and row.endswith("|")
