"""
Narrator Agent (Stage 6, optional).

Generates per-shot narration text via DeepSeek, synthesizes it with Edge-TTS,
and time-stretches the audio so each clip's narration ends within ±5% of the
clip duration.

Edge-TTS requires an internet connection. If `tts.enabled = false`, the
orchestrator skips this stage entirely.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.agents import BaseAgent
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
)
from src.utils import get_logger
from src.utils.llm_client import LLMClient
from src.utils.prompt_templates import NARRATOR_SYSTEM, NARRATOR_USER
from src.utils.video_utils import adjust_audio_speed

log = get_logger("narrator")


class NarratorAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.llm = LLMClient(config.deepseek)

    def process(self, state: PipelineState) -> PipelineState:
        if state.script_plan is None:
            raise RuntimeError("Narrator needs a script_plan")

        shots = state.all_shots()
        # Generate narration lines for every shot in one LLM call
        lines = self._generate_narration_batch(state, shots)

        for shot in shots:
            text = lines.get(shot.id)
            if not text:
                log.warning("No narration generated for %s", shot.id)
                continue
            shot.narration_text = text
            try:
                audio_path = self._synthesize_and_align(state, shot, text)
                shot.audio_clip_path = str(audio_path)
            except Exception as e:
                log.error("TTS failed for %s: %s", shot.id, e)

        state.advance_to(PipelineStage.NARRATING)
        return state

    # ---- LLM batch call -------------------------------------------------

    def _generate_narration_batch(
        self, state: PipelineState, shots: list[Shot]
    ) -> dict[str, str]:
        assert state.script_plan is not None

        shot_lines = []
        for s in shots:
            shot_lines.append(
                f"  - {s.id} ({s.duration_sec:.1f}s): {s.description}"
            )
        prompt = NARRATOR_USER.format(
            title=state.script_plan.title,
            mood=state.script_plan.overall_mood,
            shots_block="\n".join(shot_lines),
        )
        try:
            raw = self.llm.chat_json(prompt, system_prompt=NARRATOR_SYSTEM)
        except Exception as e:
            log.warning("Narration JSON call failed (%s); falling back to plain", e)
            text = self.llm.chat(prompt, system_prompt=NARRATOR_SYSTEM)
            raw = self._extract_json(text)

        result: dict[str, str] = {}
        for line in raw.get("lines", []):
            sid = line.get("shot_id")
            txt = line.get("text", "").strip()
            if sid and txt:
                result[sid] = txt
        return result

    # ---- TTS + time alignment -------------------------------------------

    def _synthesize_and_align(
        self, state: PipelineState, shot: Shot, text: str
    ) -> Path:
        import edge_tts

        raw_path = Path(state.audio_dir) / f"{shot.id}_raw.mp3"
        adj_path = Path(state.audio_dir) / f"{shot.id}.wav"

        async def _synth() -> None:
            communicate = edge_tts.Communicate(text, self.config.tts.voice)
            await communicate.save(str(raw_path))

        asyncio.run(_synth())

        return adjust_audio_speed(
            raw_path,
            target_duration_sec=shot.duration_sec,
            output_path=adj_path,
            speed_clamp=self.config.tts.speed_clamp,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON in: {text[:200]}")
        return json.loads(text[start : end + 1])
