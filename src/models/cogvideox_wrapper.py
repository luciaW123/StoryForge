"""
CogVideoX-5b-I2V inference wrapper.

Generates a short video (6 seconds at 8fps = 49 frames) starting from a
reference first frame produced by FLUX. This replaces the RIFE-based
keyframe-interpolation approach with a true video diffusion model.

Why I2V (image-to-video) and not T2V:
  - FLUX-generated keyframes have already passed the CLIP consistency gate
  - Using them as the first frame anchors visual identity across shots
  - The text prompt drives the motion, the image drives the appearance

VRAM: ~20-24 GB peak with bf16 + model_cpu_offload. Fits on a 48 GB PRO 6000
with headroom; would also fit on a 4090 with sequential_cpu_offload (slower).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils import get_logger

log = get_logger("cogvideox")


class CogVideoXWrapper:
    """Singleton-style wrapper around CogVideoXImageToVideoPipeline."""

    def __init__(self, config):
        self.config = config
        self._pipe = None
        self._offloaded = False

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import CogVideoXImageToVideoPipeline

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[getattr(self.config, "dtype", "bfloat16")]

        log.info("Loading CogVideoX I2V: %s (dtype=%s)", self.config.model_id, self.config.dtype)
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )

        # Offload strategy. PRO 6000 has 48 GB → model offload is plenty.
        offload = getattr(self.config, "offload_mode", "model")
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
            log.info("CogVideoX: sequential CPU offload (slow, low VRAM)")
        elif offload == "model":
            pipe.enable_model_cpu_offload()
            log.info("CogVideoX: model CPU offload")
        else:
            pipe.to("cuda")
            log.info("CogVideoX: fully on GPU")

        # VAE memory optimizations — important; the VAE decode for video is big
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            log.info("CogVideoX: VAE tiling + slicing enabled")
        except Exception as e:
            log.debug("VAE tiling unavailable: %s", e)

        self._pipe = pipe
        log.info("CogVideoX loaded.")

    def _ensure_on_gpu(self) -> None:
        """Bring the pipeline back to GPU if a previous `offload()` parked
        it on CPU. Mirrors FluxWrapper._ensure_on_gpu()."""
        if self._pipe is None or not self._offloaded:
            return
        offload_mode = getattr(self.config, "offload_mode", "model")
        log.info("Reloading CogVideoX pipeline from CPU to GPU (offload_mode=%s)", offload_mode)
        if offload_mode in ("model", "sequential"):
            pass  # diffusers' offload hooks still in place
        else:
            self._pipe.to("cuda")
        self._offloaded = False

    def generate(
        self,
        prompt: str,
        first_frame_path: Path,
        output_path: Path,
        seed: int = 42,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> Path:
        """Generate a video starting from `first_frame_path`, save as MP4."""
        self._load()
        self._ensure_on_gpu()
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video

        # CogVideoX-5b-I2V expects 720x480 input image
        target_w = getattr(self.config, "width", 720)
        target_h = getattr(self.config, "height", 480)
        img = Image.open(first_frame_path).convert("RGB").resize((target_w, target_h))

        gen = torch.Generator(device="cuda").manual_seed(int(seed))

        log.info(
            "Generating %d frames for prompt: %s...",
            num_frames or self.config.num_frames,
            prompt[:60],
        )

        out = self._pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=num_inference_steps or self.config.num_inference_steps,
            num_frames=num_frames or self.config.num_frames,
            guidance_scale=guidance if guidance is not None else self.config.guidance_scale,
            generator=gen,
        )
        video_frames = out.frames[0]  # list of PIL Images

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(video_frames, str(output_path), fps=self.config.fps)

        torch.cuda.empty_cache()
        return output_path

    def offload(self):
        if self._pipe is None:
            return
        try:
            self._pipe.to("cpu")
            import torch
            torch.cuda.empty_cache()
            self._offloaded = True
            log.info("CogVideoX offloaded to CPU.")
        except Exception as e:
            log.warning("Failed to offload CogVideoX: %s", e)


_singleton: Optional[CogVideoXWrapper] = None


def get_cogvideox(config) -> CogVideoXWrapper:
    global _singleton
    if _singleton is None:
        _singleton = CogVideoXWrapper(config)
    return _singleton
