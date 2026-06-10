"""
Cinematographer Agent (Stage 2 + refinement helper).

Two roles:
  1. process(): walks every shot in the ScriptPlan and fills in flux_prompt,
     negative_prompt, num_keyframes.
  2. refine_prompt(): called by the orchestrator's retry loop when the
     Consistency Agent rejects a shot. Rewrites the prompt to address the
     specific failure.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from src.agents import BaseAgent
from src.schemas.data_models import (
    Character,
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
)
from src.utils import get_logger
from src.utils.llm_client import LLMClient
from src.utils.prompt_templates import (
    CINEMATOGRAPHER_REFINE,
    CINEMATOGRAPHER_SYSTEM,
    CINEMATOGRAPHER_USER,
)

log = get_logger("cinematographer")


class CinematographerAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.llm = LLMClient(config.deepseek)

    # ---- Stage 2: bulk planning -----------------------------------------

    def process(self, state: PipelineState) -> PipelineState:
        if state.script_plan is None:
            raise RuntimeError("Cinematographer requires script_plan to be set")

        for scene in state.script_plan.scenes:
            for shot in scene.shots:
                if shot.flux_prompt:
                    continue  # already planned (e.g. resumed run)
                self._plan_shot(state, shot)

        state.advance_to(PipelineStage.PLANNING)
        return state

    def _plan_shot(self, state: PipelineState, shot: Shot) -> None:
        assert state.script_plan is not None
        scene = state.get_scene(shot.scene_id)
        if scene is None:
            raise RuntimeError(f"Shot {shot.id} references unknown scene {shot.scene_id}")

        chars = [
            state.get_character(cid)
            for cid in shot.characters_present
        ]
        chars = [c for c in chars if c is not None]
        char_block = self._format_characters(chars)

        # CogVideoX I2V flow needs exactly ONE reference keyframe per shot;
        # the motion comes from the video diffusion model, not from FLUX
        # frame interpolation. The legacy RIFE path computed a much larger
        # number from duration*fps/interp_factor (e.g. 31 frames for 5 s @
        # 24 fps / factor=4) — that wasted ~30× FLUX calls per shot and
        # delayed Stage 4 until all of them finished, which looked like
        # "consistency was skipped".
        num_kf = 1

        user_prompt = CINEMATOGRAPHER_USER.format(
            shot_description=shot.description,
            setting=scene.setting,
            time_of_day=scene.time_of_day,
            mood=scene.mood,
            character_block=char_block or "  (none)",
            camera_angle=shot.camera_angle,
            camera_movement=shot.camera_movement,
            art_style=state.script_plan.art_style,
            palette=state.script_plan.color_palette,
            duration=shot.duration_sec,
            num_keyframes=num_kf,
        )

        try:
            raw = self.llm.chat_json(user_prompt, system_prompt=CINEMATOGRAPHER_SYSTEM)
        except Exception as e:
            log.warning("Shot %s: JSON call failed (%s); falling back to plain", shot.id, e)
            text = self.llm.chat(user_prompt, system_prompt=CINEMATOGRAPHER_SYSTEM)
            raw = self._extract_json(text)

        flux_prompt = str(raw.get("flux_prompt", "")).strip()
        negative_prompt = str(
            raw.get("negative_prompt", "cartoon, anime, blurry, watermark, text")
        ).strip()
        # Ignore whatever the LLM returned for num_keyframes — CogVideoX
        # flow is fixed at 1 reference frame per shot.
        out_num_kf = 1

        # Append global style suffix (single source of truth for style tokens)
        suffix = state.script_plan.global_style_suffix or ""
        if suffix and not flux_prompt.endswith(suffix):
            flux_prompt = flux_prompt.rstrip(" ,.") + suffix

        shot.flux_prompt = flux_prompt
        shot.negative_prompt = negative_prompt
        shot.num_keyframes = out_num_kf  # always 1 in the CogVideoX flow
        log.info("Planned %s: %d kf, prompt %d chars", shot.id, shot.num_keyframes, len(flux_prompt))

    # ---- Refinement (called from orchestrator retry loop) ---------------

    def refine_prompt(self, shot: Shot, feedback: str, state: PipelineState) -> str:
        """Return an improved flux_prompt addressing the given feedback."""
        assert state.script_plan is not None
        chars = [state.get_character(cid) for cid in shot.characters_present]
        chars = [c for c in chars if c is not None]
        char_block = self._format_characters(chars)

        user_prompt = CINEMATOGRAPHER_REFINE.format(
            original_prompt=shot.flux_prompt,
            feedback=feedback,
            retry=shot.retry_count + 1,
            character_block=char_block or "  (none)",
            shot_description=shot.description,
        )
        new_text = self.llm.chat(
            user_prompt,
            system_prompt=CINEMATOGRAPHER_SYSTEM,
            temperature=0.5,
        ).strip()

        # Strip wrapping quotes / fences if the model added any
        new_text = new_text.strip("`").strip()
        if new_text.startswith('"') and new_text.endswith('"'):
            new_text = new_text[1:-1]

        suffix = state.script_plan.global_style_suffix or ""
        if suffix and not new_text.endswith(suffix):
            new_text = new_text.rstrip(" ,.") + suffix
        return new_text

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _format_characters(chars: list[Optional[Character]]) -> str:
        lines = []
        for c in chars:
            if c is None:
                continue
            lines.append(f"  - {c.name}: {c.visual_prompt}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Best-effort JSON extraction from a possibly markdown-wrapped reply."""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in: {text[:200]}")
        return json.loads(text[start : end + 1])
