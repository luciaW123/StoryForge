"""
Gemini Vision client for deep consistency checking.

Used by the Consistency Agent as the Level-2 gate after CLIP similarity
passes its fast threshold. Gemini receives:
  - The current shot's first keyframe
  - The character's reference keyframe (from first approved appearance)
  - The character description + shot description

and returns a structured judgment with actionable feedback that the
Cinematographer can use to refine the prompt on retry.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from google import genai
from google.genai import types

from src.utils import get_logger

log = get_logger("gemini")


class GeminiVisionChecker:
    """
    Two-image deep consistency check using Gemini 2.5 Flash (or any
    multimodal Gemini variant). Lazy initialization.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        # ASCII-clean just to be safe (API keys are ASCII anyway)
        self.api_key = api_key.encode('ascii', 'ignore').decode('ascii')
        self.model_name = model.encode('ascii', 'ignore').decode('ascii')
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Set it in .env or "
                "consistency.gemini_enabled=false to disable."
            )
        self._client = genai.Client(api_key=self.api_key)
        log.info("Gemini Vision client ready (model=%s)", self.model_name)

    def _pil_to_part(self, img: Image.Image, mime_type: str = "image/png") -> types.Part:
        """Convert PIL Image to a Gemini Part with explicit bytes."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime_type)

    def check_character_consistency(
        self,
        current_image: Path,
        character_description: str,
        shot_description: str,
        reference_image: Optional[Path] = None,
        character_name: str = "the character",
    ) -> dict[str, Any]:
        """
        Returns:
            {
              "consistent":  bool,
              "confidence":  float in [0, 1],
              "issues":      list[str],
              "refined_prompt_hint": str,
            }
        """
        self._load()

        # Clean user strings to avoid any non-ASCII surprises
        def clean(s: str) -> str:
            return s.encode('ascii', 'ignore').decode('ascii')

        character_description = clean(character_description)
        shot_description = clean(shot_description)
        character_name = clean(character_name)

        parts: list[Any] = []

        if reference_image is not None and Path(reference_image).exists():
            parts.append(
                "REFERENCE IMAGE — this is how the character has looked in a "
                "previously approved shot. The current image must preserve "
                "the same identity, costume, color palette, and distinctive "
                "features:"
            )
            ref_img = Image.open(reference_image).convert("RGB")
            parts.append(self._pil_to_part(ref_img))
            parts.append("CURRENT GENERATED IMAGE — judge against the reference:")
        else:
            parts.append(
                "CURRENT GENERATED IMAGE — this is the first appearance of "
                "the character, judge only against the textual description:"
            )

        curr_img = Image.open(current_image).convert("RGB")
        parts.append(self._pil_to_part(curr_img))

        parts.append(
            f"""
CHARACTER ({character_name}): {character_description}
SHOT DESCRIPTION: {shot_description}

Analyze whether the character in the CURRENT image matches the description
{"and the REFERENCE image" if reference_image else ""}.
Focus specifically on identity-relevant features (face, body, costume,
distinctive colors, accessories). Stylistic variation in lighting and
background is acceptable; identity drift is not.

Respond with ONLY a JSON object, no markdown fences, in exactly this shape:
{{
  "consistent": true | false,
  "confidence": <0.0..1.0>,
  "issues": ["short specific issue 1", "short specific issue 2"],
  "refined_prompt_hint": "concrete edit instruction the prompt-writer can apply"
}}
""".strip()
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=parts,
            )
            raw = (getattr(response, "text", None) or "").strip()
            return self._parse_json(raw)
        except Exception as e:
            log.warning("Gemini Vision call failed (%s); allowing pass-through", e, exc_info=True)
            # Fail-open: don't block the pipeline if the API is unreachable
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [f"gemini_unavailable: {type(e).__name__}"],
                "refined_prompt_hint": "",
            }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("empty response")

        # Strip ```json ... ``` fences if present
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"no JSON object in response: {text[:200]}")
            text = text[start : end + 1]

        data = json.loads(text)
        return {
            "consistent": bool(data.get("consistent", True)),
            "confidence": float(data.get("confidence", 0.5)),
            "issues": list(data.get("issues") or []),
            "refined_prompt_hint": str(data.get("refined_prompt_hint") or ""),
        }


_singleton: Optional[GeminiVisionChecker] = None


def get_gemini_checker(api_key: str, model: str = "gemini-2.5-flash") -> GeminiVisionChecker:
    global _singleton
    if _singleton is None:
        _singleton = GeminiVisionChecker(api_key=api_key, model=model)
    return _singleton