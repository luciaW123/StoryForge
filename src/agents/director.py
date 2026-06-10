"""
Director Agent (Stage 1).

Turns raw story text into a fully structured `ScriptPlan`. Uses DeepSeek with
JSON-mode for reliable structured output, with up to 3 retries on parse or
validation failure.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import ValidationError

from src.agents import BaseAgent
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    ScriptPlan,
)
from src.utils import get_logger
from src.utils.llm_client import LLMClient
from src.utils.prompt_templates import DIRECTOR_SYSTEM, DIRECTOR_USER

log = get_logger("director")


class DirectorAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.llm = LLMClient(config.deepseek)

    def process(self, state: PipelineState) -> PipelineState:
        prompt = DIRECTOR_USER.format(
            story=state.story_input,
            duration=state.target_duration_sec,
            style=state.style_hint,
            max_scenes=self.config.pipeline.max_scenes,
            max_shots=self.config.pipeline.max_shots_per_scene,
        )

        last_error: str | None = None
        for attempt in range(1, 4):
            try:
                raw = self.llm.chat_json(prompt, system_prompt=DIRECTOR_SYSTEM)
                raw = self._normalize(raw, state.target_duration_sec)
                plan = ScriptPlan.model_validate(raw)
                state.script_plan = plan
                state.advance_to(PipelineStage.SCRIPTING)
                log.info(
                    "Script built: %d scenes, %d shots, %.1fs total",
                    len(plan.scenes),
                    sum(len(s.shots) for s in plan.scenes),
                    plan.total_duration_sec,
                )
                return state
            except (ValidationError, ValueError) as e:
                last_error = str(e)
                log.warning("Attempt %d failed: %s", attempt, str(e)[:300])
                prompt = (
                    prompt
                    + f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {str(e)[:600]}"
                    + "\nFix the issues and respond ONLY with valid JSON matching the schema."
                )

        raise RuntimeError(f"Director failed after 3 attempts. Last error: {last_error}")

    # ---- Normalization --------------------------------------------------

    def _normalize(self, raw: dict[str, Any], target_duration: int) -> dict[str, Any]:
        """Patch common LLM output quirks before Pydantic validation."""
        # Assign reference_seed if missing
        for c in raw.get("characters", []):
            if "reference_seed" not in c or c["reference_seed"] is None:
                c["reference_seed"] = self._seed_from_name(c.get("name", c.get("id", "")))
            else:
                c["reference_seed"] = int(c["reference_seed"]) % (2**31)

        # Rescale durations so they sum to target
        all_shots = [
            shot
            for scene in raw.get("scenes", [])
            for shot in scene.get("shots", [])
        ]
        if all_shots:
            current_sum = sum(float(s.get("duration_sec", 0)) for s in all_shots)
            if current_sum > 0 and abs(current_sum - target_duration) > 0.5:
                factor = target_duration / current_sum
                for s in all_shots:
                    s["duration_sec"] = round(float(s["duration_sec"]) * factor, 2)
                # Adjust last shot to absorb rounding drift
                actual = sum(float(s["duration_sec"]) for s in all_shots)
                all_shots[-1]["duration_sec"] = round(
                    float(all_shots[-1]["duration_sec"]) + (target_duration - actual), 2
                )
                log.info("Rescaled shot durations by ×%.3f", factor)

        raw["total_duration_sec"] = float(target_duration)

        # Ensure character lists reference valid IDs only (drop unknowns)
        char_ids = {c["id"] for c in raw.get("characters", [])}
        for scene in raw.get("scenes", []):
            for shot in scene.get("shots", []):
                shot["characters_present"] = [
                    cid for cid in shot.get("characters_present", []) if cid in char_ids
                ]

        return raw

    @staticmethod
    def _seed_from_name(name: str) -> int:
        h = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % (2**31)
