import json
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest
import torch

from interpolate import default_output_path, is_scene_cut, main, parse_args
from rife.checkpoint import save_checkpoint
from rife.model import RIFE
from rife.video import FrameWriter, probe, read_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.manual_seed(0)
    save_checkpoint(path, RIFE(distill=False, refine=False), None, 0, 0.0, {"refine": False})
    return path


def make_video(path, size="160x96", frames=10, fps=10, audio=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    cmd += ["-frames:v", str(frames)]
    # ffv1 in .mkv accepts odd sizes; H.264 in .mp4 needs even ones.
    cmd += ["-c:v", "ffv1"] if path.suffix == ".mkv" else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return path


def make_video_from_frames(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "ffv1", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


def inspect(path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,nb_read_frames,r_frame_rate,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(res.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    return {
        "frames": int(video["nb_read_frames"]),
        "fps": Fraction(video["r_frame_rate"]),
        "size": (video["width"], video["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in streams),
    }


def run_cli(src, out, ckpt, *extra):
    return main([str(src), "--ckpt", str(ckpt), "--out", str(out), "--device", "cpu", *extra])


@pytest.mark.parametrize("factor", [2, 4])
def test_frame_count_and_fps(tmp_path, ckpt, factor):
    src = make_video(tmp_path / "in.mp4")
    assert run_cli(src, tmp_path / "out.mp4", ckpt, "--factor", str(factor)) == 0
    info = inspect(tmp_path / "out.mp4")
    assert info["frames"] == (10 - 1) * factor + 1
    assert info["fps"] == 10 * factor
    assert info["size"] == (160, 96)


def test_audio_is_copied(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mp4", audio=True)
    assert run_cli(src, tmp_path / "out.mp4", ckpt) == 0
    assert inspect(tmp_path / "out.mp4")["has_audio"]


def test_video_without_audio(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mp4", audio=False)
    assert run_cli(src, tmp_path / "out.mp4", ckpt) == 0
    assert not inspect(tmp_path / "out.mp4")["has_audio"]


def test_odd_resolution_video(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mkv", size="161x97")
    assert run_cli(src, tmp_path / "out.mkv", ckpt) == 0
    info = inspect(tmp_path / "out.mkv")
    assert info["frames"] == 19
    assert info["size"] == (162, 98)  # padded by one pixel for H.264's 4:2:0 chroma


def test_two_frame_video(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mp4", frames=2)
    assert run_cli(src, tmp_path / "out.mp4", ckpt, "--factor", "4") == 0
    assert inspect(tmp_path / "out.mp4")["frames"] == 5


def test_single_frame_video(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mp4", frames=1)
    assert run_cli(src, tmp_path / "out.mp4", ckpt) == 0
    assert inspect(tmp_path / "out.mp4")["frames"] == 1


def test_scene_cut_duplicates_instead_of_blending(tmp_path, ckpt):
    black = np.zeros((96, 160, 3), np.uint8)
    white = np.full((96, 160, 3), 255, np.uint8)
    src = make_video_from_frames(tmp_path / "cut.mkv", [black, white])
    assert run_cli(src, tmp_path / "out.mkv", ckpt) == 0
    frames = list(read_frames(tmp_path / "out.mkv", probe(tmp_path / "out.mkv")))
    assert len(frames) == 3
    assert frames[1].mean() < 20  # a duplicate of the black frame, not a grey blend


def test_is_scene_cut():
    a = np.zeros((8, 8, 3), np.uint8)
    assert not is_scene_cut(a, a, 0.2)
    assert is_scene_cut(a, np.full_like(a, 255), 0.2)


def test_missing_checkpoint_is_a_clean_error(tmp_path, capsys):
    src = make_video(tmp_path / "in.mp4", frames=2)
    assert run_cli(src, tmp_path / "out.mp4", tmp_path / "nope.pt") == 1
    assert "checkpoint not found" in capsys.readouterr().err


def test_missing_input_is_a_clean_error(tmp_path, ckpt, capsys):
    assert run_cli(tmp_path / "missing.mp4", tmp_path / "out.mp4", ckpt) == 1
    assert "input video not found" in capsys.readouterr().err


def test_rotated_video_is_upright_and_complete(tmp_path, ckpt):
    base = make_video(tmp_path / "base.mp4")  # 160x96 coded size
    rotated = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(base),
                    "-c", "copy", str(rotated)], check=True)
    assert probe(rotated).width == 96 and probe(rotated).height == 160  # ffmpeg autorotates on decode
    assert run_cli(rotated, tmp_path / "out.mp4", ckpt) == 0
    info = inspect(tmp_path / "out.mp4")
    assert info["frames"] == 19
    assert info["size"] == (96, 160)


def test_output_is_tagged_bt709(tmp_path, ckpt):
    src = make_video(tmp_path / "in.mp4", frames=3)
    assert run_cli(src, tmp_path / "out.mp4", ckpt) == 0
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=color_space",
                          "-of", "csv=p=0", str(tmp_path / "out.mp4")], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "bt709"


def test_default_output_name_is_always_writable_container(tmp_path):
    assert default_output_path(tmp_path / "clip.webm", 2) == tmp_path / "clip_2x.mp4"
    assert default_output_path(tmp_path / "clip.avi", 4) == tmp_path / "clip_4x.mp4"
    assert default_output_path(tmp_path / "clip.mkv", 2) == tmp_path / "clip_2x.mkv"
    assert default_output_path(tmp_path / "clip.MOV", 8) == tmp_path / "clip_8x.mov"


def test_out_equal_to_input_is_refused(tmp_path, ckpt, capsys):
    src = make_video(tmp_path / "in.mp4", frames=3)
    size = src.stat().st_size
    assert run_cli(src, src, ckpt) == 1
    assert "same file" in capsys.readouterr().err
    assert src.stat().st_size == size


def test_failure_mid_stream_removes_partial_output(tmp_path, ckpt, monkeypatch):
    import interpolate

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(interpolate, "interpolate_recursive", boom)
    src = make_video(tmp_path / "in.mp4", frames=3)
    out = tmp_path / "out.mp4"
    assert run_cli(src, out, ckpt) == 1
    assert not out.exists()


def test_cuda_oom_gives_a_hint(tmp_path, ckpt, monkeypatch, capsys):
    import interpolate

    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(interpolate, "interpolate_recursive", oom)
    src = make_video(tmp_path / "in.mp4", frames=3)
    assert run_cli(src, tmp_path / "out.mp4", ckpt) == 1
    assert "--scale 0.5" in capsys.readouterr().err


def test_batch_must_be_positive():
    with pytest.raises(SystemExit):
        parse_args(["in.mp4", "--ckpt", "c.pt", "--batch", "0"])


def test_ffmpeg_failure_reports_ffmpeg_error(tmp_path, ckpt, capsys):
    # The output directory doesn't exist, so the encoder dies on start-up.
    src = make_video(tmp_path / "in.mp4", frames=3)
    assert run_cli(src, tmp_path / "missing_dir" / "out.mp4", ckpt) == 1
    err = capsys.readouterr().err
    assert "ffmpeg encoding failed" in err
    assert "closed file" not in err


def test_abort_after_failed_close_cleans_up(tmp_path):
    out = tmp_path / "out.mp4"
    writer = FrameWriter(out, 16, 16, Fraction(10))
    writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.proc.kill()  # simulate ffmpeg dying mid-stream (e.g. disk full)
    with pytest.raises(RuntimeError, match="ffmpeg encoding failed"):
        for _ in range(10_000):  # until the pipe buffer fills and the write fails
            writer.write(np.zeros((16, 16, 3), np.uint8))
        writer.close()
    writer.abort()  # must not raise after close() already ran
    writer.abort()  # and must be safe to call twice
    assert not out.exists()
