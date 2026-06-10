"""
Consistency Agent (Stage 4) — two-tier identity gate.

Level 1 — CLIP coarse score (always on):
    Per-shot mean CLIP embedding compared to each character's reference.
    Produces a similarity score and a pass/fail flag against
    `character_threshold`.

Level 2 — Vision deep check (Qwen2-VL local, or Gemini fallback):
    Always invoked when a backend is configured, regardless of CLIP outcome.
    The VLM compares the current keyframe to the character's reference and
    returns a structured judgment with an actionable `refined_prompt_hint`
    that the Cinematographer can use to rewrite the prompt on retry.

A shot is approved only if BOTH levels pass. Either failure produces a
combined feedback string that names which level failed and why — letting
the Cinematographer address the actual problem on retry, and letting the
local VLM rescue CLIP false-negatives (semantically consistent frames whose
embedding drifted) without an extra Stage-3 regeneration round-trip.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents import BaseAgent
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
)
from src.utils import get_logger

log = get_logger("consistency")


class ConsistencyAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._clip = None
        # `_vision` is a duck-typed object with .check_character_consistency(...)
        # — either a GeminiVisionChecker or a LocalVisionChecker, or None.
        self._vision = None
        self._backend = "none"

        cfg = config.consistency
        # Backward compat: if gemini_enabled is true but vision_backend isn't
        # set to something explicit, honor gemini_enabled as "gemini".
        backend = (cfg.vision_backend or "").lower().strip() or (
            "gemini" if cfg.gemini_enabled else "none"
        )

        if backend == "gemini":
            api_key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                log.warning(
                    "vision_backend=gemini but no GEMINI_API_KEY; Level-2 disabled"
                )
            else:
                from src.utils.gemini_vision import get_gemini_checker
                self._vision = get_gemini_checker(api_key, cfg.gemini_model)
                self._backend = "gemini"
                log.info("Level-2 gate: Gemini Vision (%s)", cfg.gemini_model)

        elif backend == "local":
            from src.utils.local_vision import get_local_checker
            self._vision = get_local_checker(
                cfg.local_vision_model_id,
                offload_between_calls=cfg.local_vision_offload,
            )
            self._backend = "local"
            log.info(
                "Level-2 gate: Qwen2-VL local (%s, offload=%s)",
                cfg.local_vision_model_id, cfg.local_vision_offload,
            )

        elif backend == "none":
            log.info("Level-2 gate: disabled (vision_backend=none)")

        else:
            log.warning(
                "Unknown vision_backend=%r; Level-2 disabled. "
                "Valid: gemini | local | none", backend,
            )

    # ---- Lazy CLIP loading ---------------------------------------------

    def _get_clip(self):
        if self._clip is None:
            from src.models.clip_wrapper import get_clip
            self._clip = get_clip()
        return self._clip

    # ---- Bulk path (no-op fallback) ------------------------------------

    def process(self, state: PipelineState) -> PipelineState:
        for shot in state.all_shots():
            if not shot.approved:
                passed, feedback = self.evaluate_shot(state, shot)
                if passed:
                    shot.approved = True
                else:
                    shot.failure_reasons.append(feedback)
        state.advance_to(PipelineStage.CHECKING)
        return state

    # ---- Batch evaluation (VLM-resident across all shots) -------------

    def evaluate_shots_batch(
        self, state: PipelineState, shots: list[Shot]
    ) -> list[tuple[bool, str]]:
        """
        Evaluate multiple shots while keeping the local VLM resident.

        Amortizes the ~17 s Qwen-72B-AWQ reload cost across the batch by:
          1. Freeing FLUX + CogVideoX once at the start (not per-shot).
          2. Temporarily flipping the local checker's offload_between_calls
             to False so it stays resident across all `check_character_*`.
          3. Offloading the VLM exactly once after the batch completes.

        Returns: list of (passed, feedback) parallel to `shots`. Caller is
        responsible for setting `shot.approved` / `shot.failure_reasons`.

        With Gemini backend (no VRAM contention), this is just a loop with
        no special VRAM handling — same speed as per-shot mode.
        """
        if not shots:
            return []

        # Ablation gate-off: record embeddings, approve everything.
        if not self.config.consistency.enabled:
            results = []
            for shot in shots:
                self._check_characters_clip(state, shot)
                results.append((True, "gate_disabled"))
            return results

        is_local = self._backend == "local" and self._vision is not None
        prev_offload_flag = None

        if is_local:
            # Free generators ONCE for the whole batch.
            self._free_generators()
            # Keep VLM resident until batch end.
            prev_offload_flag = getattr(self._vision, "offload_between_calls", True)
            self._vision.offload_between_calls = False
            log.info(
                "consistency batch: %d shots, VLM resident",
                len(shots),
            )

        results: list[tuple[bool, str]] = []
        try:
            for shot in shots:
                results.append(self.evaluate_shot(state, shot, _skip_free=is_local))
        finally:
            if is_local:
                # Restore flag + explicit offload to release VRAM for Stage 5.
                self._vision.offload_between_calls = prev_offload_flag
                try:
                    self._vision.offload()
                except Exception as e:
                    log.warning("post-batch VLM offload failed: %s", e)

        return results

    # ---- Per-shot evaluation -------------------------------------------

    def evaluate_shot(
        self,
        state: PipelineState,
        shot: Shot,
        _skip_free: bool = False,
    ) -> tuple[bool, str]:
        """
        Returns (passed, feedback). Feedback is the raw string consumed by
        Cinematographer.refine_prompt on retry; we make it structured and
        actionable when Gemini is enabled.
        """
        if not shot.keyframe_paths:
            return False, "NO_KEYFRAMES: nothing to evaluate"

        # Ablation gate-off: still record the reference embedding for CLIP-I
        # eval comparison, but unconditionally approve the shot.
        if not self.config.consistency.enabled:
            self._check_characters_clip(state, shot)   # side-effect: sets clip_embedding
            return True, "gate_disabled"

        threshold = self.config.consistency.character_threshold

        # ---- Level 1: CLIP coarse score ----
        char_score, char_msg = self._check_characters_clip(state, shot)
        shot.consistency_score = char_score
        clip_passed = char_score >= threshold

        # ---- Level 2: vision deep check — always run when configured ----
        vision_passed = True
        vision_msg = ""
        if self._vision is not None and state.script_plan is not None:
            # Qwen2-VL-72B-AWQ needs ~39 GB VRAM and cannot coexist with FLUX
            # + CogVideoX. Free both before invoking the local VLM; they will
            # be lazily reloaded by the next stage that needs them.
            # Batch mode pre-frees once and sets _skip_free=True to avoid the
            # per-shot offload/reload churn.
            if self._backend == "local" and not _skip_free:
                self._free_generators()
            vision_passed, vision_msg = self._check_with_vision(state, shot)

        if clip_passed and vision_passed:
            tail = f", {self._backend} ok" if self._vision is not None else ""
            return True, f"OK (clip={char_score:.3f}{tail})"

        # Build combined feedback naming each failing level.
        parts: list[str] = []
        if not clip_passed:
            parts.append(
                f"CLIP_FAIL (sim={char_score:.3f} < τ={threshold}): {char_msg}"
            )
        if not vision_passed:
            tag = "GEMINI_FAIL" if self._backend == "gemini" else "VISION_FAIL"
            parts.append(f"{tag}: {vision_msg}")
        if not clip_passed and vision_passed:
            parts.append("Strengthen character visual descriptions in prompt.")
        return False, " | ".join(parts)

    # ---- Level 1 internals ---------------------------------------------

    def _check_characters_clip(
        self, state: PipelineState, shot: Shot
    ) -> tuple[float, str]:
        """
        Compute mean CLIP embedding for this shot, compare to each
        character's frozen reference. Also: on FIRST approval, set the
        reference embedding AND store the keyframe path for Gemini's use.
        """
        clip = self._get_clip()
        kf_paths = [Path(p) for p in shot.keyframe_paths]
        embeddings = [clip.encode_image(p) for p in kf_paths]
        if not embeddings:
            return 0.0, "no embeddings"

        mean_emb = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 0:
            mean_emb = mean_emb / norm

        scores: list[float] = []
        feedback_parts: list[str] = []

        if not shot.characters_present:
            # Self-consistency across this shot's frames
            internal_scores = [clip.cosine_similarity(e, mean_emb) for e in embeddings]
            score = float(np.mean(internal_scores)) if internal_scores else 1.0
            return score, f"no characters; internal coherence {score:.3f}"

        for cid in shot.characters_present:
            char = state.get_character(cid)
            if char is None:
                continue
            if char.clip_embedding is None:
                # First appearance — set both the reference embedding AND the path
                char.clip_embedding = mean_emb.tolist()
                char.reference_image_path = str(kf_paths[0])
                scores.append(1.0)
                feedback_parts.append(f"{char.name}: reference set")
                log.info(
                    "[%s] reference set for %s (img=%s)",
                    shot.id, char.name, kf_paths[0].name
                )
                continue

            ref = np.array(char.clip_embedding, dtype=np.float32)
            sim = clip.cosine_similarity(mean_emb, ref)
            scores.append(sim)
            feedback_parts.append(f"{char.name}={sim:.3f}")

        agg = float(np.mean(scores)) if scores else 1.0
        return agg, "; ".join(feedback_parts)

    # ---- Level 2 internals: vision backend (Gemini or local) -----------

    def _check_with_vision(
        self, state: PipelineState, shot: Shot
    ) -> tuple[bool, str]:
        """
        The active vision backend gets the current keyframe + (when
        available) the character's reference keyframe + the character & shot
        descriptions, and returns a structured judgment. We forward
        refined_prompt_hint as the feedback string so the Cinematographer
        can act on it directly.
        """
        assert self._vision is not None
        if not shot.characters_present:
            log.info(
                "[%s] vision SKIPPED — shot has no characters_present "
                "(Director did not assign any). Treating as pass.",
                shot.id,
            )
            return True, "no character present; vision skipped"

        current = Path(shot.keyframe_paths[0])

        char_descs = []
        reference_image: Optional[Path] = None
        primary_name = "the character"
        for cid in shot.characters_present:
            c = state.get_character(cid)
            if c is None:
                continue
            char_descs.append(f"{c.name}: {c.visual_prompt}")
            if reference_image is None and c.reference_image_path:
                ref_path = Path(c.reference_image_path)
                if ref_path.exists() and ref_path.resolve() != current.resolve():
                    reference_image = ref_path
                    primary_name = c.name

        character_description = " | ".join(char_descs) if char_descs else "unspecified"

        verdict = self._vision.check_character_consistency(
            current_image=current,
            character_description=character_description,
            shot_description=shot.description,
            reference_image=reference_image,
            character_name=primary_name,
        )

        consistent = bool(verdict.get("consistent", True))
        conf = float(verdict.get("confidence", 0.5))
        issues = verdict.get("issues") or []
        hint = verdict.get("refined_prompt_hint", "") or ""

        if consistent:
            return True, f"{self._backend} ok (conf={conf:.2f})"

        # Build feedback Cinematographer can act on
        issues_str = "; ".join(str(x) for x in issues[:3]) if issues else "unspecified drift"
        feedback = (
            f"identity drift (conf={conf:.2f}): {issues_str}. "
            f"HINT: {hint}" if hint else
            f"identity drift (conf={conf:.2f}): {issues_str}."
        )
        return False, feedback

    # ---- VRAM management ----------------------------------------------

    def _free_generators(self) -> None:
        """
        Offload FLUX and CogVideoX from GPU before invoking the local VLM.
        The Qwen2-VL-72B-AWQ needs ~39 GB resident; on a 96 GB GPU with
        FLUX (~24 GB) and CogVideoX (~28 GB) already on-device the VLM load
        would OOM. We free them here; they will be lazily reloaded by their
        respective wrappers (singletons) on the next call.
        """
        # FLUX
        try:
            from src.models.flux_wrapper import _singleton as flux_singleton
            if flux_singleton is not None:
                flux_singleton.offload()
        except Exception as e:
            log.debug("flux offload skipped: %s", e)

        # CogVideoX
        try:
            from src.models.cogvideox_wrapper import _singleton as cog_singleton
            if cog_singleton is not None:
                cog_singleton.offload()
        except Exception as e:
            log.debug("cogvideox offload skipped: %s", e)

        # Best-effort GPU cleanup
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
