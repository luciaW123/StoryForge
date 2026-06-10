"""
Visual Agent (Stage 3).

Generates N keyframe images per shot using FLUX.1-dev. Supports both bulk
processing via `process(state)` and per-shot generation via
`generate_shot(state, shot)` — the latter is used by the orchestrator's
retry loop.

Seed strategy:
  - Solo-character static shot: same seed for every frame (max consistency)
  - Solo-character motion shot: incremental seeds (subtle drift = motion)
  - Multi-character or no-character: hash(shot.id) base + incremental
"""

from __future__ import annotations

import hashlib
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

log = get_logger("visual")


class VisualAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._flux = None  # lazy

    # ---- Lazy FLUX loading ----------------------------------------------

    def _get_flux(self):
        if self._flux is None:
            from src.models.flux_wrapper import get_flux

            self._flux = get_flux(self.config.flux)
        return self._flux

    # ---- Bulk path (process all unfinished shots) -----------------------

    def process(self, state: PipelineState) -> PipelineState:
        if state.script_plan is None:
            raise RuntimeError("Visual agent needs a planned script")

        for shot in state.all_shots():
            if not shot.approved and not shot.keyframe_paths:
                self.generate_shot(state, shot)

        state.advance_to(PipelineStage.GENERATING)
        return state

    # ---- Per-shot generation (called by orchestrator retry loop) --------

    def generate_shot(self, state: PipelineState, shot: Shot) -> Shot:
        flux = self._get_flux()
        kf_dir = Path(state.keyframes_dir) / shot.id
        kf_dir.mkdir(parents=True, exist_ok=True)

        # Clean previous attempts from this shot's directory
        for old in kf_dir.glob("*.png"):
            old.unlink()

        seeds = self._seeds_for_shot(state, shot)
        paths: list[str] = []

        # Describe where the seed came from, so the log makes it obvious that
        # retries REUSE the same deterministic seed (not a fresh random one).
        char_ids = shot.characters_present
        if len(char_ids) == 1:
            seed_src = f"char {char_ids[0]}.reference_seed"
        elif len(char_ids) == 0:
            seed_src = "hash(shot.id)"
        else:
            seed_src = f"combine({sorted(char_ids)})"

        log.info(
            "Generating %d keyframes for %s (attempt=%d, steps=%d, %dx%d) "
            "seeds=%s [%s]",
            shot.num_keyframes,
            shot.id,
            shot.retry_count + 1,
            self.config.flux.num_inference_steps,
            self.config.flux.width,
            self.config.flux.height,
            seeds,
            seed_src,
        )

        for i, seed in enumerate(seeds):
            out_path = kf_dir / f"{shot.id}_kf{i:03d}.png"
            try:
                flux.generate_to_file(
                    prompt=shot.flux_prompt,
                    output_path=out_path,
                    seed=seed,
                    negative_prompt=shot.negative_prompt,
                )
                paths.append(str(out_path))
            except Exception as e:
                log.error("Frame %d of %s failed: %s", i, shot.id, e)
                raise

        shot.keyframe_paths = paths
        log.info("Shot %s: %d keyframes written", shot.id, len(paths))
        # Binary-search debug: dump the FLUX keyframe BEFORE it reaches Animator.
        if paths:
            dump_artifact(state, shot, "flux_keyframe.png", paths[0])
        return shot

    # ---- Seed strategy --------------------------------------------------

    def _seeds_for_shot(self, state: PipelineState, shot: Shot) -> list[int]:
        n = shot.num_keyframes
        char_ids = shot.characters_present

        if len(char_ids) == 1:
            char = state.get_character(char_ids[0])
            base = char.reference_seed if char else self._hash_seed(shot.id)
        elif len(char_ids) == 0:
            base = self._hash_seed(shot.id)
        else:
            # Sort char_ids so seed is order-independent across shots.
            # Combine each character's own reference_seed so that Dr. Volkov
            # always contributes the same value whether he's solo or in a group.
            chars = [state.get_character(cid) for cid in sorted(char_ids)]
            char_seeds = [c.reference_seed for c in chars if c is not None]
            if char_seeds:
                base = char_seeds[0]
                for s in char_seeds[1:]:
                    base = (base * 1000003 + s) % (2**31)
            else:
                base = self._hash_seed(shot.id)

        static = shot.camera_movement == "static"
        if static and len(char_ids) <= 1:
            # Identical seed → near-identical frames (RIFE will still interpolate
            # smoothly; tiny FLUX denoise variance prevents perfectly frozen output)
            return [base] * n
        # Otherwise, drift the seed to suggest natural motion
        return [(base + i) % (2**31) for i in range(n)]

    @staticmethod
    def _hash_seed(key: str) -> int:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % (2**31)
