"""
DeepSeek API client.

DeepSeek exposes an OpenAI-compatible Chat Completions endpoint, so we just
configure the `openai` client with the DeepSeek base URL and a DeepSeek key.
Two helpers cover all agent needs:

  - chat()         : plain text completion
  - chat_json()    : forces JSON response (uses `response_format`)
  - chat_vision()  : sends a base64-encoded image with the message

All calls retry on transient errors (connection drops, 5xx) with exponential
backoff. The client is thread-safe; agents may be invoked concurrently.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Optional

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.schemas.data_models import DeepSeekConfig
from src.utils import get_logger

log = get_logger("llm")


_TRANSIENT_ERRORS = (APITimeoutError, APIError, RateLimitError, ConnectionError)
_INVALID_REQUEST_MARKERS = (
    "invalid_request_error",
    "unknown variant `image_url`",
    "expected `text`",
)


class VisionNotSupportedError(RuntimeError):
    """Raised when the configured model does not accept image inputs."""


def _is_invalid_request(err: Exception) -> bool:
    msg = str(err)
    return any(marker in msg for marker in _INVALID_REQUEST_MARKERS)


class LLMClient:
    """Thin DeepSeek wrapper with retries and JSON-mode helper."""

    def __init__(self, config: DeepSeekConfig, max_retries: int = 5):
        if not config.api_key:
            raise RuntimeError(
                "DeepSeek API key is empty. Set DEEPSEEK_API_KEY in the environment."
            )
        self.config = config
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60.0,
        )

    # ---- Core calls ------------------------------------------------------

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._call(
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model or self.config.model,
        )

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Force JSON output. Raises ValueError if response is not parseable."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        raw = self._call(
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=self.config.model,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.error("JSON parse failure: %s\nRaw: %s", e, raw[:500])
            raise ValueError(f"LLM returned non-JSON: {e}") from e

    def chat_vision(
        self,
        user_prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a single image plus text. Uses base64 inline encoding."""
        img_b64 = _encode_image_b64(Path(image_path))
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        )
        try:
            return self._call(
                messages=messages,
                temperature=0.3,
                max_tokens=512,
                model=self.config.vision_model,
            )
        except RuntimeError as e:
            if _is_invalid_request(e):
                raise VisionNotSupportedError(
                    "Vision model does not support image inputs. "
                    "Set deepseek.vision_model to a vision-capable model "
                    "or disable consistency.use_deepseek_vision."
                ) from e
            raise

    # ---- Internals -------------------------------------------------------

    def _call(self, **kwargs: Any) -> str:
        delay = 1.0
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except _TRANSIENT_ERRORS as e:
                if _is_invalid_request(e):
                    raise RuntimeError(str(e)) from e
                last_err = e
                log.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt == self.max_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError(f"LLM call exhausted retries: {last_err}") from last_err


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
