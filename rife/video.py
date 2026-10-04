"""Stream frames through ffmpeg pipes: no temporary files, flat memory for any video length."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    has_audio: bool
    variable_frame_rate: bool


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH; install it (e.g. `sudo apt install ffmpeg`)")


def _fraction(text: str) -> Fraction:
    num, _, den = text.partition("/")
    den = den or "1"
    return Fraction(int(num), int(den)) if int(den) else Fraction(0)


def _rotation(stream: dict) -> float:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return float(side["rotation"])
    try:
        return float(stream.get("tags", {}).get("rotate", 0))  # legacy tag
    except ValueError:
        return 0.0


def probe(path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input video not found: {path}")
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
         "-show_entries", "stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise ValueError(f"could not read {path}: {res.stderr.strip()}")
    streams = json.loads(res.stdout).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    r_rate = _fraction(video.get("r_frame_rate", "0/1"))
    avg_rate = _fraction(video.get("avg_frame_rate", "0/1"))
    fps = avg_rate if avg_rate > 0 else r_rate
    if fps <= 0:
        raise ValueError(f"could not determine the frame rate of {path}")
    vfr = r_rate > 0 and avg_rate > 0 and abs(r_rate - avg_rate) / r_rate > 0.01
    width, height = int(video["width"]), int(video["height"])
    # ffmpeg autorotates on decode, so the pixels it pipes out have the displayed (rotated) size,
    # not the coded size ffprobe reports. Phone portrait clips are the common case.
    if abs(_rotation(video)) % 180 == 90:
        width, height = height, width
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        variable_frame_rate=vfr,
    )


def read_frames(path, info: VideoInfo) -> Iterator[np.ndarray]:
    # -map 0:v:0 makes the decoded stream the same one probe() measured.
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = info.width * info.height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class FrameWriter:
    def __init__(self, path, width: int, height: int, fps: Fraction, audio_source=None, crf: int = 18):
        self.path = Path(path)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        if audio_source is not None:
            # "1:a?" = copy audio if the source has any; duration is unchanged so it stays in sync.
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        filters = []
        if width % 2 or height % 2:
            # H.264 with 4:2:0 chroma needs even dimensions; pad one pixel rather than crop.
            filters.append(f"pad={width + width % 2}:{height + height % 2}")
        # ffmpeg's default RGB->YUV matrix is BT.601, but players assume BT.709 for HD; convert
        # with 709 explicitly and tag the stream so colours match the source.
        filters.append("scale=out_color_matrix=bt709:out_range=tv")
        cmd += ["-vf", ",".join(filters)]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close()  # raises with ffmpeg's error message
            raise

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:  # ffmpeg already died; its stderr below says why
            pass
        self.proc.stdin = None  # communicate() would otherwise try to flush the closed pipe
        _, err = self.proc.communicate()  # also closes the stderr pipe
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {err.decode(errors='replace').strip()}")

    def abort(self) -> None:
        """Stop encoding and delete the unusable partial output. Safe after close()."""
        try:
            # close() may already have reaped ffmpeg and closed its pipes; a second
            # communicate() would then fail with "read of closed file".
            if self.proc.returncode is None:
                self.proc.kill()
                if self.proc.stdin:
                    try:
                        self.proc.stdin.close()
                    except BrokenPipeError:
                        pass
                    self.proc.stdin = None
                self.proc.communicate()
        finally:
            self.path.unlink(missing_ok=True)
