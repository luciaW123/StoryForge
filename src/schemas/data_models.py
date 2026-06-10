"""
StoryForge Data Models
======================

All structured data that flows between agents. Every agent reads from and
writes to a single shared `PipelineState`. Models are validated on assignment
so a malformed update from any agent fails fast at the source.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """State machine stages. Order matters — stages advance monotonically."""

    INIT = "INIT"
    SCRIPTING = "SCRIPTING"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    CHECKING = "CHECKING"
    INTERPOLATING = "INTERPOLATING"
    NARRATING = "NARRATING"
    EDITING = "EDITING"
    DONE = "DONE"
    FAILED = "FAILED"

    @classmethod
    def ordered(cls) -> list["PipelineStage"]:
        return [
            cls.INIT,
            cls.SCRIPTING,
            cls.PLANNING,
            cls.GENERATING,
            cls.CHECKING,
            cls.INTERPOLATING,
            cls.NARRATING,
            cls.EDITING,
            cls.DONE,
        ]

    def is_terminal(self) -> bool:
        return self in (PipelineStage.DONE, PipelineStage.FAILED)

    def next(self) -> "PipelineStage":
        order = self.ordered()
        idx = order.index(self)
        return order[idx + 1] if idx + 1 < len(order) else PipelineStage.DONE


class TransitionType(str, Enum):
    CUT = "cut"
    FADE_TO_BLACK = "fade_to_black"
    DISSOLVE = "dissolve"
    FADE_IN = "fade_in"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye level"
    LOW_ANGLE = "low angle"
    HIGH_ANGLE = "high angle"
    BIRDS_EYE = "birds eye"
    DUTCH_ANGLE = "dutch angle"
    OVER_THE_SHOULDER = "over the shoulder"
    CLOSE_UP = "close up"
    MEDIUM_SHOT = "medium shot"
    WIDE_SHOT = "wide shot"
    EXTREME_WIDE = "extreme wide"


class CameraMovement(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan left"
    PAN_RIGHT = "pan right"
    TILT_UP = "tilt up"
    TILT_DOWN = "tilt down"
    ZOOM_IN = "zoom in"
    ZOOM_OUT = "zoom out"
    PUSH_IN = "push in"
    PULL_OUT = "pull out"
    DOLLY = "dolly"
    HANDHELD = "handheld"


# ---------------------------------------------------------------------------
# Story-level models
# ---------------------------------------------------------------------------


class Character(BaseModel):
    """A persistent character across the story. `clip_embedding` is populated
    after the first approved appearance and used as the reference for
    consistency checks in later shots."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    name: str
    description: str
    visual_prompt: str = Field(
        ...,
        description="FLUX-optimized visual description, e.g. 'a red fox with "
        "vibrant amber fur, white-tipped bushy tail'",
    )
    reference_seed: int = Field(..., ge=0, lt=2**31)
    clip_embedding: Optional[list[float]] = None
    reference_image_path: Optional[str] = None  # set on first approved shot; used by Gemini Vision

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not v.startswith("char_"):
            raise ValueError("Character id must start with 'char_'")
        return v


class Shot(BaseModel):
    """A single continuous camera shot. The atomic unit of generation:
    one shot → N keyframes → 1 interpolated video clip + optional audio."""

    model_config = ConfigDict(validate_assignment=True)

    # Identity
    id: str
    scene_id: str
    order: int = Field(..., ge=0)
    description: str

    # Visual prompt (filled by Cinematographer)
    flux_prompt: str = ""
    negative_prompt: str = ""

    # Cinematography
    camera_angle: str = CameraAngle.EYE_LEVEL.value
    camera_movement: str = CameraMovement.STATIC.value
    transition_out: str = TransitionType.CUT.value

    # Timing
    duration_sec: float = Field(..., gt=0.0, le=30.0)
    num_keyframes: int = Field(default=0, ge=0)

    # Character involvement
    characters_present: list[str] = Field(default_factory=list)

    # Execution outputs (filled during pipeline run)
    keyframe_paths: list[str] = Field(default_factory=list)
    video_clip_path: Optional[str] = None
    audio_clip_path: Optional[str] = None
    narration_text: Optional[str] = None

    # Quality tracking
    consistency_score: float = 0.0
    approved: bool = False
    retry_count: int = 0
    failure_reasons: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if "_sh" not in v:
            raise ValueError("Shot id should follow pattern '<scene>_sh<n>'")
        return v

    def expected_keyframes(self, fps: int = 24, interp_factor: int = 4) -> int:
        """Number of keyframes required to produce a smooth clip at the
        target framerate after RIFE interpolation by `interp_factor`."""
        return math.ceil(self.duration_sec * fps / interp_factor) + 1


class Scene(BaseModel):
    """A narrative unit containing one or more shots that share setting/mood."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    order: int = Field(..., ge=0)
    title: str
    setting: str
    time_of_day: str
    mood: str
    description: str
    shots: list[Shot]

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not v.startswith("scene_"):
            raise ValueError("Scene id must start with 'scene_'")
        return v

    @model_validator(mode="after")
    def _shots_belong(self) -> "Scene":
        for s in self.shots:
            if s.scene_id != self.id:
                raise ValueError(
                    f"Shot {s.id} has scene_id={s.scene_id!r} "
                    f"but belongs to scene {self.id!r}"
                )
        return self


class ScriptPlan(BaseModel):
    """Top-level screenplay produced by the Director and refined by the
    Cinematographer. The single source of truth for narrative structure."""

    model_config = ConfigDict(validate_assignment=True)

    title: str
    genre: str
    overall_mood: str
    art_style: str
    global_style_suffix: str = Field(
        ...,
        description="Style tokens appended to every FLUX prompt for visual "
        "coherence, e.g. ', cinematic lighting, 35mm film, photorealistic, 8k'",
    )
    color_palette: str
    characters: list[Character]
    scenes: list[Scene]
    total_duration_sec: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _validate_duration_sum(self) -> "ScriptPlan":
        actual = sum(shot.duration_sec for scene in self.scenes for shot in scene.shots)
        if abs(actual - self.total_duration_sec) > 0.5:
            raise ValueError(
                f"Sum of shot durations ({actual:.2f}s) does not match "
                f"total_duration_sec ({self.total_duration_sec:.2f}s)"
            )
        return self

    @model_validator(mode="after")
    def _character_refs_exist(self) -> "ScriptPlan":
        char_ids = {c.id for c in self.characters}
        for scene in self.scenes:
            for shot in scene.shots:
                for cid in shot.characters_present:
                    if cid not in char_ids:
                        raise ValueError(
                            f"Shot {shot.id} references unknown character {cid!r}"
                        )
        return self


# ---------------------------------------------------------------------------
# Pipeline state (mutable, passed between agents)
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """The single object that flows through all agents. Serializable to JSON
    for checkpointing — every field must be JSON-friendly."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str
    stage: PipelineStage = PipelineStage.INIT

    # Original user input
    story_input: str
    target_duration_sec: int = Field(..., gt=0, le=300)
    style_hint: str = "cinematic, photorealistic"

    # Filled by Director (Stage 1)
    script_plan: Optional[ScriptPlan] = None

    # File system layout
    output_dir: str
    keyframes_dir: str
    clips_dir: str
    audio_dir: str
    final_video_path: Optional[str] = None

    # Metadata
    created_at: str
    updated_at: str

    # Statistics (for end-of-run summary)
    total_shots: int = 0
    approved_shots: int = 0
    total_retries: int = 0
    stage_durations_sec: dict[str, float] = Field(default_factory=dict)

    # ---- Factory ---------------------------------------------------------

    @classmethod
    def new(
        cls,
        story: str,
        duration: int,
        style: str,
        output_root: Path | str = "outputs",
    ) -> "PipelineState":
        """Create a fresh state with a unique session id and provisioned dirs."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        root = Path(output_root) / session_id
        keyframes = root / "keyframes"
        clips = root / "clips"
        audio = root / "audio"
        for d in (root, keyframes, clips, audio):
            d.mkdir(parents=True, exist_ok=True)
        now = datetime.now().isoformat()
        return cls(
            session_id=session_id,
            story_input=story,
            target_duration_sec=duration,
            style_hint=style,
            output_dir=str(root),
            keyframes_dir=str(keyframes),
            clips_dir=str(clips),
            audio_dir=str(audio),
            created_at=now,
            updated_at=now,
        )

    # ---- Convenience accessors ------------------------------------------

    def all_shots(self) -> list[Shot]:
        """All shots in global execution order."""
        if self.script_plan is None:
            return []
        result: list[Shot] = []
        for scene in sorted(self.script_plan.scenes, key=lambda s: s.order):
            for shot in sorted(scene.shots, key=lambda sh: sh.order):
                result.append(shot)
        return result

    def get_character(self, char_id: str) -> Optional[Character]:
        if self.script_plan is None:
            return None
        for c in self.script_plan.characters:
            if c.id == char_id:
                return c
        return None

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        if self.script_plan is None:
            return None
        for s in self.script_plan.scenes:
            if s.id == scene_id:
                return s
        return None

    # ---- State transitions ----------------------------------------------

    def advance_to(self, new_stage: PipelineStage) -> None:
        """Move forward to a later stage (monotonic)."""
        order = PipelineStage.ordered()
        if new_stage in order and self.stage in order:
            if order.index(new_stage) < order.index(self.stage):
                raise ValueError(
                    f"Cannot regress from {self.stage} to {new_stage}"
                )
        self.stage = new_stage
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, reason: str) -> None:
        self.stage = PipelineStage.FAILED
        self.updated_at = datetime.now().isoformat()
        # Append failure info to script_plan if it exists, else create a marker
        self.final_video_path = None


# ---------------------------------------------------------------------------
# Configuration (loaded from config.yaml)
# ---------------------------------------------------------------------------


class DeepSeekConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    vision_model: str = "deepseek-chat"  # reserved for vision-capable variant


class FluxConfig(BaseModel):
    model_id: str = "black-forest-labs/FLUX.1-dev"
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 576
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = True
    offload_mode: Optional[str] = None    # "none"|"model"|"sequential"; overrides enable_cpu_offload


class RIFEConfig(BaseModel):
    executable: str = "rife-ncnn-vulkan/rife-ncnn-vulkan.exe"
    model: str = "rife-v4.6"
    interpolation_factor: int = 4
    gpu_id: int = 0


class CogVideoXConfig(BaseModel):
    """Configuration for CogVideoX image-to-video motion synthesizer."""
    model_id: str = "THUDM/CogVideoX-5b-I2V"
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    num_frames: int = 49        # 49 frames at 8 fps = 6.125 seconds per shot
    fps: int = 8
    width: int = 720            # CogVideoX-5b-I2V expects 720×480
    height: int = 480
    dtype: str = "bfloat16"
    offload_mode: Optional[str] = "model"   # "none"|"model"|"sequential"


class TTSConfig(BaseModel):
    enabled: bool = True
    voice: str = "en-US-GuyNeural"
    speed_clamp: tuple[float, float] = (0.75, 1.25)


class VideoConfig(BaseModel):
    target_fps: int = 24
    output_codec: str = "libx264"
    output_bitrate: str = "8000k"
    width: int = 1920
    height: int = 1080


class ConsistencyConfig(BaseModel):
    # Master switch — set False for ablation "gate off" baseline.
    # All shots are auto-approved; clip_embedding references are still recorded
    # (first-appearance logic in _check_characters_clip still runs) so that
    # eval.py can compute CLIP-I on the gate-off run for comparison.
    enabled: bool = True

    # Level 1: CLIP fast filter. Frames whose similarity to character reference
    # is below this hard floor are rejected immediately without invoking Gemini.
    character_threshold: float = 0.80
    scene_threshold: float = 0.70
    max_retries_per_shot: int = 3

    # Legacy DeepSeek-vision check — kept off by default; weak because
    # DeepSeek-chat does not have a true vision modality.
    use_deepseek_vision: bool = False

    # Level 2: deep vision check. Triggered after CLIP passes. Provides
    # structured feedback (refined_prompt_hint) for Cinematographer rewrite.
    #
    #   vision_backend = "gemini"  -> Gemini 2.x Flash via google-genai (cloud)
    #   vision_backend = "local"   -> Qwen2-VL-7B-Instruct via transformers (GPU)
    #   vision_backend = "none"    -> skip Level 2 entirely (CLIP-only gating)
    vision_backend: str = "gemini"

    # Legacy single-flag; only respected when vision_backend == "gemini".
    # New code should set vision_backend = "none" to fully disable.
    gemini_enabled: bool = True
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""              # set via ${GEMINI_API_KEY} in config.yaml

    # Qwen2-VL-72B local backend. Default Level-2 gate. ~40 GB resident.
    # Default is the GPTQ-Int4 snapshot (works on cu130 + Blackwell via
    # gptqmodel); AWQ was tried first but AutoAWQ's Triton kernel is broken
    # on this hardware. Cannot coexist with FLUX + CogVideoX, so the agent
    # calls _free_generators() before each invocation and the checker
    # deletes itself afterwards when offload_between_calls=True.
    local_vision_model_id: str = "/root/autodl-tmp/hf_cache/qwen2vl-72b-gptq-int4"
    local_vision_offload: bool = True     # delete model after each call to free VRAM

    # Batch mode: generate ALL keyframes first (FLUX resident), then free FLUX
    # and check ALL keyframes in one VLM session (Qwen resident), amortizing
    # the 17 s VLM reload across N shots. Trade-off: retries are also batched,
    # so a single bad shot will not block the others' first pass.
    # Set false to use the legacy per-shot interleaving (debug-friendly, slow).
    batch_mode: bool = True


class PipelineSettings(BaseModel):
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    max_scenes: int = 8
    max_shots_per_scene: int = 3
    log_level: str = "INFO"


class PipelineConfig(BaseModel):
    """Aggregate configuration. Loaded once at startup from `config.yaml`."""

    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    flux: FluxConfig = Field(default_factory=FluxConfig)
    rife: RIFEConfig = Field(default_factory=RIFEConfig)
    cogvideox: CogVideoXConfig = Field(default_factory=CogVideoXConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        import os

        import yaml

        raw = Path(path).read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(raw) or {}

        # Expand ${ENV_VAR} references in deepseek.api_key (and any string value)
        def _expand(obj: Any) -> Any:
            if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                return os.environ.get(obj[2:-1], "")
            if isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand(v) for v in obj]
            return obj

        return cls.model_validate(_expand(data))
