"""
Image-generation backend wrapper.

Despite the historical name `FluxWrapper`, this class now supports BOTH:
  - black-forest-labs/FLUX.1-dev      (24 GB transformer — needs sequential offload on 24 GB GPUs; slow)
  - black-forest-labs/FLUX.1-schnell  (same architecture; 4-step distilled)
  - stabilityai/stable-diffusion-xl-base-1.0  (7 GB — fits comfortably with model offload; fast)
  - any other StableDiffusionXLPipeline-compatible checkpoint

The pipeline class is selected by inspecting `config.model_id`. The agent-facing
interface (`generate`, `generate_to_file`, `offload`) is unchanged so no caller
needs to be modified.

On a 24 GB RTX 4090, SDXL is the recommended backend:
  - FLUX-dev needs sequential offload (~30-100 s per keyframe)
  - SDXL with model offload runs at ~1 s/step (~25 s per keyframe at 25 steps)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from src.schemas.data_models import FluxConfig
from src.utils import get_logger

log = get_logger("flux")


def _is_flux_model(model_id: str) -> bool:
    return "flux" in model_id.lower()


class FluxWrapper:
    """
    Backend-agnostic image generator. Name kept for backward compatibility;
    handles both FLUX and SDXL pipelines internally.
    """

    def __init__(self, config: FluxConfig):
        self.config = config
        self._pipe = None
        self._is_flux = _is_flux_model(config.model_id)
        self._offloaded = False  # True while pipe lives on CPU after offload()

    # ---- Loading -------------------------------------------------------

    def _load(self):
        if self._pipe is not None:
            return
        import torch

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.dtype]

        log.info(
            "Loading %s pipeline: %s (dtype=%s)",
            "FLUX" if self._is_flux else "SDXL",
            self.config.model_id,
            self.config.dtype,
        )

        if self._is_flux:
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(self.config.model_id, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                add_watermarker=False,
            )

        # Apply offload mode
        offload_mode = getattr(self.config, "offload_mode", None) or (
            "model" if self.config.enable_cpu_offload else "none"
        )
        if offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload()
            log.info("Offload mode: sequential (low VRAM, slow)")
        elif offload_mode == "model":
            pipe.enable_model_cpu_offload()
            log.info("Offload mode: model (moderate VRAM, fast)")
        else:
            pipe.to("cuda")
            log.info("Offload mode: none (max VRAM, fastest)")

        # VAE tiling — reduces peak memory during decode at high resolutions
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            log.info("VAE tiling + slicing enabled")
        except Exception as e:
            log.debug("VAE tiling unavailable: %s", e)

        self._pipe = pipe
        log.info("Pipeline loaded.")

    # ---- Generation ----------------------------------------------------

    def _ensure_on_gpu(self) -> None:
        """If the pipe was previously offloaded to CPU (e.g. by the
        Consistency agent's `_free_generators()`), bring it back to GPU
        before the next generation. No-op when offload_mode is
        model/sequential (diffusers' own offload wrappers manage placement)."""
        if self._pipe is None or not self._offloaded:
            return
        offload_mode = getattr(self.config, "offload_mode", None) or (
            "model" if self.config.enable_cpu_offload else "none"
        )
        log.info("Reloading FLUX pipeline from CPU to GPU (offload_mode=%s)", offload_mode)
        if offload_mode in ("model", "sequential"):
            # Re-applying enable_*_cpu_offload on the same pipe is unsafe;
            # diffusers' offload hooks survived the .to('cpu'), so calling
            # generate() will move submodules back as needed. Nothing to do.
            pass
        else:
            self._pipe.to("cuda")
        self._offloaded = False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> Image.Image:
        self._load()
        self._ensure_on_gpu()
        import torch

        gen = torch.Generator("cuda").manual_seed(int(seed))
        w = width or self.config.width
        h = height or self.config.height
        n_steps = steps or self.config.num_inference_steps
        cfg = guidance if guidance is not None else self.config.guidance_scale

        if self._is_flux:
            # FLUX: no native negative prompt; uses max_sequence_length for T5
            out = self._pipe(
                prompt=prompt,
                num_inference_steps=n_steps,
                guidance_scale=cfg,
                width=w,
                height=h,
                generator=gen,
                max_sequence_length=512,
            )
        else:
            # SDXL: negative prompt supported
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=n_steps,
                guidance_scale=cfg,
                width=w,
                height=h,
                generator=gen,
            )

        # Free fragmented allocations between keyframes
        torch.cuda.empty_cache()
        return out.images[0]

    def generate_to_file(
        self,
        prompt: str,
        output_path: Path,
        seed: int = 0,
        negative_prompt: str = "",
    ) -> Path:
        img = self.generate(prompt=prompt, negative_prompt=negative_prompt, seed=seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    # ---- Cleanup -------------------------------------------------------

    def offload(self) -> None:
        """Move pipeline to CPU to free VRAM for other models (CLIP / VLM /
        CogVideoX). Mark `_offloaded` so the next `generate()` knows to
        re-move tensors back to GPU."""
        if self._pipe is None:
            return
        try:
            self._pipe.to("cpu")
            import torch
            torch.cuda.empty_cache()
            self._offloaded = True
            log.info("Pipeline offloaded to CPU.")
        except Exception as e:
            log.warning("Failed to offload pipeline: %s", e)


_singleton: Optional[FluxWrapper] = None


def get_flux(config: FluxConfig) -> FluxWrapper:
    global _singleton
    if _singleton is None:
        _singleton = FluxWrapper(config)
    return _singleton
