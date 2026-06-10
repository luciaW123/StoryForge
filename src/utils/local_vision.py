"""
Qwen2-VL-72B-Instruct-GPTQ-Int4 local vision checker.

Local backend for the Consistency Agent's Level-2 deep check. Same interface
as `GeminiVisionChecker.check_character_consistency()` so `consistency.py`
only needs a backend-selection branch.

Model: Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4
  - GPTQ 4-bit weights (~40 GB resident VRAM)
  - Loaded with torch_dtype="auto" and device_map="auto"
  - DO NOT pass a `quantization_config` to from_pretrained — GPTQ weights are
    already quantized; transformers reads the embedded config from the
    model's config.json.

VRAM strategy: the 72B GPTQ model cannot coexist with FLUX + CogVideoX during
a single shot's evaluation. The Consistency Agent calls `_free_generators()`
before each Level-2 invocation to unload them, and calls `offload()` on this
checker after the call to release VRAM back to the generators.

Path: /root/autodl-tmp/hf_cache/qwen2vl-72b-gptq-int4
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.utils import get_logger

log = get_logger("local_vision")


# Resolved at import time. Override via env:
#     export QWEN2VL_MODEL_PATH=/some/other/snapshot
# Falls back to the standard autodl path otherwise.
_FALLBACK_MODEL_PATH = "/root/autodl-tmp/hf_cache/qwen2vl-72b-gptq-int4"
DEFAULT_MODEL_PATH = os.environ.get("QWEN2VL_MODEL_PATH", _FALLBACK_MODEL_PATH)


class LocalVisionChecker:
    """
    Qwen2-VL-72B vision checker. Lazy initialization.

    Interface matches `GeminiVisionChecker.check_character_consistency`:
    same kwargs, same return-dict keys
    (`consistent`, `confidence`, `issues`, `refined_prompt_hint`).

    Lifecycle:
      _load()    — load weights to GPU (device_map="auto")
      offload()  — delete model, free VRAM (call after each consistency check
                   so FLUX/CogVideoX can reload)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        offload_between_calls: bool = True,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.offload_between_calls = offload_between_calls
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    # ---- Lifecycle -----------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # Block all HF Hub network calls. local_files_only=True cannot be used
        # with absolute local paths on huggingface-hub==0.30.0 — the library
        # validates repo-id format BEFORE checking os.path.isdir, so
        # "/root/..." triggers "Repo id must be in the form 'namespace/repo'".
        # The env var is checked AFTER the isdir gate, so it works correctly.
        os.environ["HF_HUB_OFFLINE"] = "1"

        log.info("Loading Qwen2-VL-72B from %s ...", self.model_id)
        # GPTQ weights: do NOT pass quantization_config; transformers reads the
        # embedded GPTQ config from the snapshot's config.json.
        # torch_dtype="auto" honors each tensor's on-disk dtype.
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        # min_pixels / max_pixels bound the vision encoder cost
        # 256*28*28 .. 1280*28*28 ≈ 200K .. 1M pixels per image
        min_px = 256 * 28 * 28
        max_px = 1280 * 28 * 28

        # The Qwen2-VL-72B snapshot ships a preprocessor_config.json whose
        # `size` field is in an older shape (e.g. {"height": ..., "width": ...})
        # that transformers >= 4.51 rejects with "size must contain
        # 'shortest_edge' and 'longest_edge' keys". We override `size` to the
        # canonical shape on load; this also ignores whatever the snapshot had.
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=min_px,
                max_pixels=max_px,
                size={"shortest_edge": min_px, "longest_edge": max_px},
            )
        except (ValueError, TypeError) as e:
            log.warning(
                "AutoProcessor.from_pretrained failed (%s); "
                "falling back to manual Qwen2VLImageProcessor + tokenizer wiring",
                e,
            )
            from transformers import (
                AutoTokenizer,
                Qwen2VLImageProcessor,
                Qwen2VLProcessor,
            )

            image_processor = Qwen2VLImageProcessor(
                min_pixels=min_px,
                max_pixels=max_px,
                size={"shortest_edge": min_px, "longest_edge": max_px},
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._processor = Qwen2VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        log.info(
            "Qwen2-VL-72B ready (offload_between_calls=%s)",
            self.offload_between_calls,
        )

    def offload(self) -> None:
        """
        Fully release VRAM. AWQ + device_map='auto' weights cannot simply be
        moved to CPU with .to('cpu') (some submodules are dispatched), so we
        delete the model object and run GC + cuda.empty_cache.

        After offload, the next check_character_consistency() call reloads
        the weights from disk (~10-15 s warm cache).
        """
        if self._model is None:
            return
        try:
            import torch
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log.info("Qwen2-VL-72B offloaded (model deleted, VRAM freed).")
        except Exception as e:
            log.warning("Failed to offload Qwen2-VL: %s", e)

    # ---- Inference -----------------------------------------------------

    def check_character_consistency(
        self,
        current_image: Path,
        character_description: str,
        shot_description: str,
        reference_image: Optional[Path] = None,
        character_name: str = "the character",
    ) -> dict[str, Any]:
        """Returns the same dict shape as GeminiVisionChecker."""
        self._load()
        import torch

        # Build the multimodal message in Qwen2-VL's chat format
        images: list[Image.Image] = []
        content_blocks: list[dict[str, Any]] = []

        if reference_image is not None and Path(reference_image).exists():
            content_blocks.append({"type": "text", "text":
                "REFERENCE IMAGE - this is how the character has looked in a "
                "previously approved shot. The current image must preserve "
                "the same identity, costume, color palette, and distinctive "
                "features."
            })
            ref_img = Image.open(reference_image).convert("RGB")
            images.append(ref_img)
            content_blocks.append({"type": "image", "image": ref_img})
            content_blocks.append({"type": "text", "text":
                "CURRENT GENERATED IMAGE - judge against the reference:"
            })
        else:
            content_blocks.append({"type": "text", "text":
                "CURRENT GENERATED IMAGE - this is the first appearance of "
                "the character; judge only against the textual description:"
            })

        curr_img = Image.open(current_image).convert("RGB")
        images.append(curr_img)
        content_blocks.append({"type": "image", "image": curr_img})

        instruction = f"""
CHARACTER ({character_name}): {character_description}
SHOT DESCRIPTION: {shot_description}

Analyze whether the character in the CURRENT image matches the description{
    " and the REFERENCE image" if reference_image else ""
}. Focus on identity-relevant features (face, body, costume, distinctive
colors, accessories). Stylistic variation in lighting/background is fine;
identity drift is not.

Respond with ONLY a JSON object, no markdown, in exactly this shape:
{{
  "consistent": true | false,
  "confidence": <0.0..1.0>,
  "issues": ["short specific issue 1", "short specific issue 2"],
  "refined_prompt_hint": "concrete edit instruction the prompt-writer can apply"
}}
""".strip()
        content_blocks.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content_blocks}]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            # device_map="auto" placed the embed layer on a specific device;
            # move inputs to the same device as the model's input embeddings.
            try:
                input_device = next(self._model.parameters()).device
            except StopIteration:
                input_device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(input_device)

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            prompt_len = inputs.input_ids.shape[1]
            output_ids = generated_ids[0, prompt_len:]
            raw = self._processor.decode(output_ids, skip_special_tokens=True).strip()

            return self._parse_json(raw)
        except Exception as e:
            # Loud — silent pass-through hid a Triton/dtype kernel crash for
            # multiple pipeline runs. Surface the failure prominently so the
            # operator notices, and mark the verdict as inconsistent (rather
            # than "consistent=True" pass-through) so the orchestrator at
            # least logs a retry attempt instead of silently approving.
            log.error(
                "Qwen2-VL call FAILED (%s: %s) — VLM verdict unavailable; "
                "returning consistent=False so the failure is visible",
                type(e).__name__, e, exc_info=True,
            )
            return {
                "consistent": False,
                "confidence": 0.0,
                "issues": [f"local_vision_unavailable: {type(e).__name__}: {e}"],
                "refined_prompt_hint": "",
            }
        finally:
            if self.offload_between_calls:
                self.offload()

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


_singleton: Optional[LocalVisionChecker] = None


def get_local_checker(
    model_id: str = DEFAULT_MODEL_PATH,
    offload_between_calls: bool = True,
) -> LocalVisionChecker:
    global _singleton
    if _singleton is None:
        _singleton = LocalVisionChecker(model_id, offload_between_calls)
    return _singleton
