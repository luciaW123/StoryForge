"""Frame/video helpers used by Animator, Narrator, and Post agents."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2

from src.utils import get_logger

log = get_logger("video_utils")


def frames_to_mp4(
    frames: Iterable[Path],
    output_path: Path,
    fps: int = 24,
    codec: str = "mp4v",
) -> Path:
    """Encode an ordered list of PNG frames into an MP4 clip using OpenCV."""
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                log.warning("Skipping unreadable frame: %s", f)
                continue
            writer.write(img)
    finally:
        writer.release()
    return output_path


def get_video_duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return float(frames) / float(fps) if fps > 0 else 0.0
    finally:
        cap.release()


def get_audio_duration_sec(path: Path) -> float:
    """Use ffprobe to read audio duration. Requires ffmpeg installed."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def adjust_audio_speed(
    audio_path: Path,
    target_duration_sec: float,
    output_path: Path,
    speed_clamp: tuple[float, float] = (0.75, 1.25),
) -> Path:
    """Stretch or compress audio to match target duration using ffmpeg atempo."""
    actual = get_audio_duration_sec(audio_path)
    if actual <= 0:
        raise RuntimeError(f"Audio has zero duration: {audio_path}")
    factor = actual / target_duration_sec
    factor = max(speed_clamp[0], min(speed_clamp[1], factor))
    if abs(factor - 1.0) < 0.02:
        # close enough, just copy
        shutil.copyfile(audio_path, output_path)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            f"atempo={factor:.4f}",
            "-loglevel",
            "error",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def copy_frames_flat(frames: Iterable[Path], dest_dir: Path) -> Path:
    """Copy frames to a flat directory with sequential numeric names.
    Required by rife-ncnn-vulkan which expects 00000001.png ... format."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(sorted(frames), start=1):
        target = dest_dir / f"{i:08d}.png"
        shutil.copyfile(f, target)
    return dest_dir
