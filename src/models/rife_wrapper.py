"""
RIFE frame interpolation wrapper.

We call the pre-compiled `rife-ncnn-vulkan` binary as a subprocess. The binary
takes a directory of input frames and writes an interpolated sequence to an
output directory.

CLI reference:
  rife-ncnn-vulkan -i <input_dir> -o <output_dir> -m <model> -n <factor> -g 0
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.schemas.data_models import RIFEConfig
from src.utils import get_logger

log = get_logger("rife")


class RIFEWrapper:
    def __init__(self, config: RIFEConfig):
        self.config = config
        self.exe = Path(config.executable)
        if not self.exe.exists():
            log.warning(
                "RIFE executable not found at %s. Animator stage will fail at runtime. "
                "Download from https://github.com/nihui/rife-ncnn-vulkan/releases",
                self.exe,
            )

    def interpolate(
        self,
        input_dir: Path,
        output_dir: Path,
        factor: int | None = None,
    ) -> Path:
        """Run RIFE on a directory of frames and return the output directory."""
        if not self.exe.exists():
            raise FileNotFoundError(f"RIFE executable not found: {self.exe}")

        factor = factor or self.config.interpolation_factor
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-m", self.config.model,
            "-n", str(factor),
            "-g", str(self.config.gpu_id),
            "-j", "4:4:4",
            "-f", "%08d.png",
        ]
        log.info("Running RIFE: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("RIFE stderr: %s", result.stderr[-1000:])
            raise RuntimeError(f"RIFE failed (exit {result.returncode})")
        return output_dir

    @staticmethod
    def expected_output_count(input_frame_count: int, factor: int) -> int:
        """RIFE 4x on K frames -> 4K-3 frames (interpolates between pairs)."""
        if input_frame_count < 2:
            return input_frame_count
        return input_frame_count + (input_frame_count - 1) * (factor - 1)
