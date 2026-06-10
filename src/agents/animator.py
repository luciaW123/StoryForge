"""
Animator Agent (Stage 5) — CogVideoX-driven motion synthesis.

For each approved shot:
  1. Free FLUX from GPU (it occupies VRAM we'll need for CogVideoX).
  2. Use the first FLUX keyframe (the one that passed the CLIP consistency
     gate) as the conditioning image.
  3. Run CogVideoX-5b-I2V with the shot's text prompt to generate a
     6-second video starting from that frame.
  4. Save the resulting MP4 as shot.video_clip_path.

This replaces the earlier RIFE-based interpolation. RIFE was a poor fit
because it requires real motion between adjacent frames; FLUX's independently
generated keyframes don't have that. CogVideoX is a true video diffusion
model — it synthesizes coherent motion directly, conditioned on a starting
image and a text prompt.

Architecture story:
  FLUX → CLIP gate (identity)  →  CogVideoX (motion) → cross-dissolve assembly

The CLIP consistency check still serves its role: it approves the FLUX
keyframe before that frame is handed to CogVideoX as the visual anchor.
This means consistency at the FRAME-IDENTITY level is enforced; motion
coherence within the shot is handled by CogVideoX's temporal attention.

Fallback paths (env vars):
  FORCE_RIFE=1     → use RIFE (needs Vulkan, broken in most containers)
  FORCE_STILLS=1   → render each shot as a static still (no motion at all)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from src.agents import BaseAgent
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
)
from src.utils import get_logger
from src.utils.debug_dump import dump_artifact
from src.utils.video_utils import copy_frames_flat, frames_to_mp4

log = get_logger("animator")


class AnimatorAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._rife = None
        self._cogvideox = None

    # ---- Lazy model accessors ------------------------------------------

    def _get_rife(self):
        if self._rife is None:
            from src.models.rife_wrapper import RIFEWrapper
            self._rife = RIFEWrapper(self.config.rife)
        return self._rife

    def _get_cogvideox(self):
        if self._cogvideox is None:
            from src.models.cogvideox_wrapper import get_cogvideox
            self._cogvideox = get_cogvideox(self.config.cogvideox)
        return self._cogvideox

    def _backend(self) -> str:
        if os.environ.get("FORCE_RIFE", "0") == "1":
            return "rife"
        if os.environ.get("FORCE_STILLS", "0") == "1":
            return "stills"
        return "cogvideox"

    # ---- Main loop -----------------------------------------------------

    def process(self, state: PipelineState) -> PipelineState:
        backend = self._backend()
        log.info("Animator backend: %s", backend)

        # On low-VRAM systems, free FLUX before loading CogVideoX.
        # On 100 GB+ GPUs, both can coexist — skip the offload to save load time.
        free_flux = os.environ.get("OFFLOAD_FLUX", "0") == "1"
        if backend == "cogvideox" and free_flux:
            self._try_offload_flux()

        for shot in state.all_shots():
            if not shot.approved:
                log.warning("Skipping unapproved shot %s", shot.id)
                continue
            if shot.video_clip_path and Path(shot.video_clip_path).exists():
                log.info("Shot %s already done; skipping", shot.id)
                continue
            self._make_clip(state, shot, backend)

        state.advance_to(PipelineStage.INTERPOLATING)
        return state

    def _try_offload_flux(self):
        """Best-effort offload of FLUX from VRAM before CogVideoX loads."""
        try:
            from src.models.flux_wrapper import _singleton as flux_singleton
            if flux_singleton is not None:
                flux_singleton.offload()
        except Exception as e:
            log.debug("FLUX offload skipped: %s", e)

    def _make_clip(self, state: PipelineState, shot: Shot, backend: str) -> Path:
        kf_paths = [Path(p) for p in shot.keyframe_paths]
        if not kf_paths:
            raise RuntimeError(f"Shot {shot.id} has no keyframes")

        if backend == "cogvideox":
            return self._build_with_cogvideox(state, shot, kf_paths[0])
        if backend == "rife" and len(kf_paths) >= 2:
            return self._build_with_rife(state, shot, kf_paths)
        return self._build_with_still(state, shot, kf_paths[0])

    # ---- CogVideoX path (DEFAULT) --------------------------------------

    def _build_with_cogvideox(
        self, state: PipelineState, shot: Shot, first_frame: Path
    ) -> Path:
        cog = self._get_cogvideox()
        clip_path = Path(state.clips_dir) / f"{shot.id}.mp4"

        # Use the cinematographer's prompt; fall back to the description.
        # Strip the FLUX-specific global_style_suffix before passing to CogVideoX —
        # tokens like "35mm film" / "photorealistic, 8k" are misread by CogVideoX
        # as visual-effect triggers and produce scan lines / full-screen flashing.
        prompt = (shot.flux_prompt or shot.description or "").strip()
        if state.script_plan:
            suffix = (state.script_plan.global_style_suffix or "").strip()
            if suffix and prompt.endswith(suffix):
                prompt = prompt[: -len(suffix)].rstrip(" ,.")
        if not prompt:
            prompt = shot.description or ""
        if not prompt:
            raise RuntimeError(f"Shot {shot.id} has no prompt for CogVideoX")
        # Append motion-safe hints instead of FLUX aesthetic tokens
        prompt = prompt.rstrip(" ,.") + ", smooth natural motion, no visual effects, no flashing lights, no glitch artifacts"
        print("[DEBUG] CogVideoX prompt:", prompt)

        # Deterministic seed from shot id for reproducibility
        seed = abs(hash(shot.id)) % (2**31)

        cog.generate(
            prompt=prompt,
            first_frame_path=first_frame,
            output_path=clip_path,
            seed=seed,
        )

        shot.video_clip_path = str(clip_path)
        log.info("Shot %s: CogVideoX clip → %s", shot.id, clip_path.name)
        # Binary-search debug: dump the raw CogVideoX clip BEFORE Post.
        dump_artifact(state, shot, "cogvideo_raw.mp4", clip_path)
        return clip_path

    # ---- Static-still fallback -----------------------------------------

    def _build_with_still(
        self, state: PipelineState, shot: Shot, image: Path
    ) -> Path:
        fps = self.config.video.target_fps
        duration = shot.duration_sec if shot.duration_sec > 0 else 2.0
        W = self.config.video.width
        H = self.config.video.height
        clip_path = Path(state.clips_dir) / f"{shot.id}.mp4"
        clip_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(fps),
            "-t", f"{duration:.3f}",
            "-i", str(image),
            "-vf", f"scale={W}:{H}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {shot.id}: {result.stderr[-300:]}")
        shot.video_clip_path = str(clip_path)
        log.info("Shot %s: %.2fs still (fallback)", shot.id, duration)
        return clip_path

    # ---- RIFE fallback (FORCE_RIFE=1, needs Vulkan) --------------------

    def _build_with_rife(
        self, state: PipelineState, shot: Shot, kf_paths: list[Path]
    ) -> Path:
        rife = self._get_rife()
        factor = self.config.rife.interpolation_factor
        fps = self.config.video.target_fps

        with tempfile.TemporaryDirectory(prefix=f"{shot.id}_rife_") as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            copy_frames_flat(kf_paths, in_dir)
            rife.interpolate(in_dir, out_dir, factor=factor)

            frames = sorted(out_dir.glob("*.png"))
            if not frames:
                raise RuntimeError(f"RIFE produced no frames for {shot.id}")

            clip_path = Path(state.clips_dir) / f"{shot.id}.mp4"
            frames_to_mp4(frames, clip_path, fps=fps)

        shot.video_clip_path = str(clip_path)
        log.info("Shot %s (RIFE): %d frames", shot.id, len(frames))
        return clip_path
