# StoryForge — Master Project Reference

> 2026 Spring Computer Graphics — Project 3
> **Topic**: Multi-Agent Pipeline for Locally-Deployable Text-to-Video Generation
> **Hardware target**: single NVIDIA RTX PRO 6000 (96 GB) or comparable workstation GPU
> **Latest revision**: 2026-05-26 — switched motion backend from RIFE to CogVideoX-5b-I2V; switched consistency precision-check backend from Gemini Vision to local Qwen2-VL

This is the single-source-of-truth design document. It supersedes the per-topic
files in `docs/` where they conflict.

---

## Table of Contents

1. [Project Context & Topic Rationale](#1-project-context--topic-rationale)
2. [Requirements Coverage Matrix](#2-requirements-coverage-matrix)
3. [System Architecture](#3-system-architecture)
4. [Data Models (Pydantic Schemas)](#4-data-models-pydantic-schemas)
5. [The Seven Agents](#5-the-seven-agents)
6. [The Core Contribution: Consistency-Gated Video Synthesis](#6-the-core-contribution-consistency-gated-video-synthesis)
7. [Pipeline Orchestration & Checkpointing](#7-pipeline-orchestration--checkpointing)
8. [Tech Stack & Dependencies](#8-tech-stack--dependencies)
9. [Configuration Reference (`config.yaml`)](#9-configuration-reference-configyaml)
10. [VRAM & Disk Budget](#10-vram--disk-budget)
11. [Runtime Performance & Timing](#11-runtime-performance--timing)
12. [File & Directory Layout](#12-file--directory-layout)
13. [CLI Usage](#13-cli-usage)
14. [Deployment](#14-deployment)
15. [Failure Modes & Recovery](#15-failure-modes--recovery)
16. [Evaluation Plan](#16-evaluation-plan)
17. [Known Limitations & Future Work](#17-known-limitations--future-work)
18. [Architectural Evolution: Why CogVideoX Replaced RIFE](#18-architectural-evolution-why-cogvideox-replaced-rife)
19. [Glossary](#19-glossary)

---

## 1. Project Context & Topic Rationale

### 1.1 The assignment

Project 3 from the 2026 Spring Computer Graphics course requires a substantial,
original implementation in one of three broad directions: **3D scene
generation**, **video generation**, or **agent-based visual content
automation**. Deliverables: working code in a public repo, a 6–8 page
academic-style report (ICLR template, PDF), a presentation deck, and a live
demo. Time budget: ~3 weeks. Team size: 1.

### 1.2 Topic choice — multi-agent text-to-video

After evaluating three candidates (poster generator, 3D scene synthesis,
text-to-video), I selected **multi-agent text-to-video**:

- The single direction where multi-agent decomposition produces genuine
  engineering and intellectual leverage (no single model handles the whole
  pipeline well)
- Highest research relevance for the 2025–26 graphics curriculum
- Most demo-impactful output (a video plays standalone)
- Tractable on a single workstation GPU given the right model choices

### 1.3 What "locally-deployable" means here

- **All generation runs on the local GPU**: FLUX.1-dev (keyframe synthesis),
  CLIP ViT-L/14 (consistency coarse-gating), Qwen2-VL (consistency precision-gating),
  CogVideoX-5b-I2V (motion synthesis). No proprietary visual model is required
  on the main path.
- **DeepSeek API** is used only for *language reasoning* — story-to-script
  decomposition, prompt engineering.
- **Gemini Vision API** is kept as an optional fallback for consistency precision
  checks (when local Qwen2-VL is unavailable); it is no longer on the primary path.
- **Edge-TTS** for narration is optional and toggleable.

The system runs on a workstation with an NVIDIA RTX PRO 6000 (96 GB) or a
single-rental cloud GPU container of comparable scale. ≥48 GB VRAM is the
minimum viable configuration (with offload); 96 GB is recommended for full
speed and full residency of FLUX + CogVideoX + Qwen2-VL together.

---

## 2. Requirements Coverage Matrix

| PDF Criterion | Weight | Concrete deliverable |
|---|---|---|
| Technical depth | 30% | 7-agent pipeline with structured shared state; CLIP-gated keyframe approval that anchors per-shot video synthesis; coordinated FLUX → CLIP → CogVideoX execution under VRAM constraints |
| Novelty / originality | 20% | **Consistency-gated cascade**: visual-identity anchoring via CLIP-approved FLUX keyframes feeding CogVideoX's image-to-video diffusion ensures character identity propagates through motion; Pydantic-typed agent protocol with per-stage checkpointing |
| Engineering quality | 20% | Type-safe data models (Pydantic v2); resumable checkpoints; graceful SIGINT handling; lazy model loading; backend-agnostic image wrapper (FLUX / SDXL auto-detect); pluggable motion backend (CogVideoX / RIFE / static) |
| Result quality | 15% | Per-shot 6-second 720×480 videos with coherent motion, assembled into 12–18 s 1080p H.264 deliverables with target CLIP-I > 0.80 across shots |
| Report & presentation | 15% | 6–8 page ICLR-format paper; figures of architecture, ablations, failure cases; live CLI demo |

---

## 3. System Architecture

### 3.1 High-level diagram

```
                          ┌──────────────────────────────┐
                          │   User: "A fox in snow..."   │
                          └──────────────┬───────────────┘
                                         ▼
              ┌────────────────────────────────────────────────────┐
              │  PipelineOrchestrator                              │
              │  (state machine + checkpoints + retry loop)        │
              └─┬──────────────────────────────────────────────────┘
                │
                ▼
   Stage 1   ┌─────────────────┐    DeepSeek-V3 (text)
   ─────────►│ Director Agent  ├─────────────────────────► ScriptPlan
             └─────────────────┘
                │
                ▼
   Stage 2   ┌──────────────────────┐  DeepSeek-V3 (text)
   ─────────►│ Cinematographer Agent├─────────────────────► Shot.flux_prompt
             └──────────────────────┘
                │
                ▼
   Stage 3   ┌─────────────────┐    FLUX.1-dev (bf16, GPU)
   ─────────►│ Visual Agent    ├─────────────────────────► Shot.keyframe_paths
             └─────────────────┘    (1 reference frame per shot)
                │
                ▼
   Stage 4   ┌─────────────────┐    CLIP ViT-L/14 (coarse) + Qwen2-VL (precision)
   ─────────►│ Consistency     ├──fail──► retry Stage 3 with refined prompt
             │ Agent           │
             └───────┬─────────┘
                     │ pass
                     ▼
   Stage 5   ┌─────────────────┐    CogVideoX-5b-I2V (bf16, GPU)
   ─────────►│ Animator Agent  ├─────────────────────────► Shot.video_clip_path
             └─────────────────┘    (6.1 s actual motion video per shot)
                │
                ▼
   Stage 6   ┌─────────────────┐    DeepSeek + Edge-TTS (optional)
   ─────────►│ Narrator Agent  ├─────────────────────────► Shot.audio_clip_path
             └─────────────────┘
                │
                ▼
   Stage 7   ┌─────────────────┐    MoviePy + ffmpeg (H.264)
   ─────────►│ Post Agent      ├─────────────────────────► final_video.mp4
             └─────────────────┘    (1920×1080, cross-dissolve transitions)
```

### 3.2 Why this decomposition

A single monolithic prompt-to-video model has three structural limitations:

1. **No shot-level control**: cannot say "shot 3 should be a close-up of the
   fox's eyes" without re-prompting the whole video.
2. **Limited duration**: most open-source T2V models cap at ~6 seconds.
3. **Character drift**: without explicit identity preservation, the protagonist's
   appearance drifts across the generated frames.

Decomposing into discrete shots and chaining specialized models solves all three:

- Each shot is independently prompt-engineered (no monolithic prompt limit).
- Total length is unbounded — we concatenate per-shot CogVideoX clips.
- CLIP-similarity on FLUX-generated *first frames* gates each shot's appearance;
  CogVideoX then renders motion *from* that approved frame, so identity
  propagates into the video.

### 3.3 The state machine

Stages execute in linear order. The only non-linear edge is **Stage 4 → Stage 3**
on consistency failure. Encoded as the `PipelineStage` enum:

```
INIT → SCRIPTING → PLANNING → GENERATING ⇆ CHECKING → INTERPOLATING → NARRATING → EDITING → DONE
                                        ↑_______↓
                                     (retry, max 3 per shot)
```

Note: stage names `GENERATING` / `INTERPOLATING` are historical — Stage 5
no longer interpolates frames, it synthesizes motion videos. The names are
preserved for backward compatibility with checkpoint files.

Each stage writes a checkpoint on completion. Resume:
`python main.py resume <session_id>`.

---

## 4. Data Models (Pydantic Schemas)

All inter-agent communication uses Pydantic v2 models in
`src/schemas/data_models.py`. Gives us: JSON serialization for checkpoints,
validation at agent boundaries, auto-generated schema documentation.

### 4.1 Enums

```python
class PipelineStage(str, Enum):
    INIT, SCRIPTING, PLANNING, GENERATING, CHECKING,
    INTERPOLATING, NARRATING, EDITING, DONE

class TransitionType(str, Enum):
    CUT, FADE_TO_BLACK, FADE_IN, DISSOLVE

class CameraAngle(str, Enum):
    WIDE, MEDIUM, CLOSE_UP, EXTREME_CLOSE_UP,
    OVER_SHOULDER, LOW_ANGLE, HIGH_ANGLE

class CameraMovement(str, Enum):
    STATIC, PAN, TILT, ZOOM_IN, ZOOM_OUT, DOLLY
```

### 4.2 Character

```python
class Character(BaseModel):
    id: str                          # "char_fox"
    name: str                        # "Fox"
    visual_prompt: str               # "red fox, white belly, alert ears..."
    reference_seed: int              # FLUX seed for consistency
    clip_embedding: Optional[list[float]] = None  # set after first approval
```

### 4.3 Shot

```python
class Shot(BaseModel):
    id: str                          # "scene_1_sh1"
    scene_id: str
    order: int
    description: str
    duration_sec: float              # ≥ 1.0
    flux_prompt: Optional[str]
    negative_prompt: str
    characters_present: list[str]
    camera_angle: CameraAngle
    camera_movement: CameraMovement
    transition_out: TransitionType
    num_keyframes: int               # default 1 in the CogVideoX flow
    keyframe_paths: list[str] = []
    video_clip_path: Optional[str]   # set by Animator (CogVideoX output)
    audio_clip_path: Optional[str]
    narration_text: Optional[str]
    consistency_score: float = 0.0
    approved: bool = False
    retry_count: int = 0
    failure_reasons: list[str] = []
```

**Important change from earlier RIFE-based design**:
`num_keyframes` is now typically 1 — the single reference frame for CogVideoX I2V.
Multiple keyframes are no longer needed because the motion comes from CogVideoX,
not from frame interpolation.

### 4.4 Scene, ScriptPlan, PipelineState, PipelineConfig

See `src/schemas/data_models.py` for full definitions. The shapes are unchanged
from earlier revisions; only the *semantics* of `Shot.keyframe_paths` shifted
(from "frames to interpolate" to "reference frame for video diffusion").

---

## 5. The Seven Agents

All agents inherit from `BaseAgent` with the minimal `process(state) → state`
contract. Stateless beyond the config they receive.

### 5.1 Director (`src/agents/director.py`)

Converts free-text story input into a structured `ScriptPlan` via a single
DeepSeek `chat_json` call. Validates output against the Pydantic schema; on
`ValidationError`, retries up to 3× with the error text appended to the prompt.
Normalizes shot durations to sum to the target.

### 5.2 Cinematographer (`src/agents/cinematographer.py`)

For each shot, generates a FLUX-compatible prompt (one call per shot to allow
prompt refinement on retry). Output: `{flux_prompt, negative_prompt, rationale}`.

In the CogVideoX flow, `num_keyframes = 1` — we only need a single anchor frame
per shot, not many keyframes for interpolation.

`refine_prompt(shot, feedback, state)` is called by the orchestrator on
consistency failure, rewriting the prompt to address specific feedback
("fox's fur color drifted — emphasize 'rust-red coat'").

### 5.3 Visual (`src/agents/visual.py`)

Generates the reference keyframe(s) for each shot using FLUX.1-dev.

| FLUX parameter | Value | Why |
|---|---|---|
| `model_id` | `black-forest-labs/FLUX.1-dev` | Best open-source image quality 2024–25 |
| `dtype` | `bfloat16` | Halves VRAM vs fp32, no quality loss |
| `enable_cpu_offload` | configurable | `none` on 96 GB; `model` on 48 GB; `sequential` on 24 GB |
| `num_inference_steps` | 20 | Sweet spot for FLUX-dev |
| `guidance_scale` | 3.5 | FLUX-recommended |
| `width × height` | 1024 × 576 | 16:9, native FLUX aspect |

Per-keyframe cost: ~5 s on 96 GB with no offload.

The wrapper (`src/models/flux_wrapper.py`) auto-detects pipeline type from
`model_id`, also supporting SDXL as a smaller alternative.

### 5.4 Consistency (`src/agents/consistency.py`)

See Section 6 for the deep dive. The agent runs a **two-stage cascade**:

**Stage 4a — CLIP coarse score (always on, fast):**
- Computes mean CLIP embedding of the shot's keyframe(s) via `open_clip` ViT-L/14.
- On first character appearance: sets the reference embedding.
- On subsequent shots: computes cosine similarity vs the frozen reference;
  produces a pass/fail flag against `character_threshold` (default 0.80).
  Result is *not* short-circuited — Stage 4b runs regardless.

**Stage 4b — Qwen2-VL precision gate (local VLM, always invoked):**
- Sends the shot's keyframe + reference frame + the textual character description
  to a local Qwen2-VL model and asks for a structured judgment
  (identity-match / attribute-drift / text-image alignment + free-text reason).
- Two model sizes are supported:
  - **Qwen2-VL-7B** (integrated and operational) — default light-tier check.
  - **Qwen2-VL-72B-AWQ** (~45 GB, download in progress, target path
    `/root/autodl-tmp/hf_cache/qwen2vl-72b-awq`) — heavy-tier check for the
    final eval runs; activated once the download completes.
- Output: structured `(passed, reasons[])` merged with the CLIP verdict.

A shot is approved only when **both** Stage 4a and Stage 4b pass; either
failure routes back to Stage 3 with combined feedback. Running both
unconditionally (rather than short-circuiting on CLIP failure) lets Qwen2-VL
rescue CLIP false-negatives — semantically faithful frames whose embedding
drifted — without an extra FLUX regeneration round-trip. The added cost is
~5–10 s per shot (7B) since the local VLM has no per-call billing.

**Optional cloud fallback:** if `consistency.vlm_backend: gemini` (or local
weights are missing), the agent falls back to the Gemini Vision API for
Stage 4b. Kept as a fallback only — not on the primary path.

Returns `(passed: bool, feedback: str)`; orchestrator decides what to do.

### 5.5 Animator (`src/agents/animator.py`) — **NOW BUILDS REAL VIDEOS**

For each approved shot, generates a 6.1-second video using CogVideoX-5b-I2V
conditioned on:
- **First frame**: the shot's FLUX-generated keyframe (CLIP-approved)
- **Text prompt**: the shot's `flux_prompt` (already cinematographically tuned)

CogVideoX parameters (defaults for 96 GB GPU):

| Parameter | Value | Note |
|---|---|---|
| `model_id` | `THUDM/CogVideoX-5b-I2V` | 5B params, image-to-video distilled |
| `dtype` | `bfloat16` | Quality-preserving half-precision |
| `offload_mode` | `none` | Full GPU residency on 96 GB |
| `num_inference_steps` | 50 | Diffusion sampling steps |
| `guidance_scale` | 6.0 | Standard for CogVideoX |
| `num_frames` | 49 | At 8 fps = 6.125 s clip |
| `width × height` | 720 × 480 | CogVideoX-5b-I2V's native size |

Per-shot cost on 96 GB with no offload: **~2.5–3 min**. With model offload on a
48 GB card: ~4 min. Total VRAM footprint with FLUX co-resident: ~50–60 GB peak
during VAE decode.

**Backend selection** (controlled by env vars):
- Default: CogVideoX I2V (motion synthesis)
- `FORCE_RIFE=1`: legacy RIFE-based interpolation (requires working Vulkan ICD)
- `STATIC_ONLY=1`: render each shot as a held still (fallback)

### 5.6 Narrator (`src/agents/narrator.py`)

Optional. Single batched DeepSeek call generates per-shot narration text;
Edge-TTS synthesizes audio; `ffmpeg atempo` stretches each line to match the
shot's actual video duration (6.125 s for CogVideoX shots). Disable via
`tts.enabled: false`.

### 5.7 Post (`src/agents/post.py`)

MoviePy assembly:
1. Load each shot's CogVideoX-generated MP4 as `VideoFileClip`.
2. Attach narration audio if present.
3. Apply per-shot outgoing transition (cut / fade-to-black / dissolve).
4. Concatenate with crossfade overlap where `transition_out == DISSOLVE`.
5. Resize to 1920 × 1080.
6. Write H.264 8 Mbps + AAC 192 kbps to `final_video.mp4`.

---

## 6. The Core Contribution: Consistency-Gated Video Synthesis

This is the central technical claim of the paper.

### 6.1 The problem

When generating video shot-by-shot, two failure modes dominate:

1. **Within-shot incoherence**: independently-generated frames don't have shared
   motion → optical-flow interpolation produces distortion.
2. **Cross-shot drift**: even if each shot is internally coherent, the
   protagonist's appearance drifts from one shot to the next.

Pure image-generation pipelines (FLUX + classical interpolation, e.g. RIFE)
solve neither well: RIFE works only when source frames represent real motion,
and FLUX alone provides no mechanism for cross-shot identity.

### 6.2 Our solution — a two-stage gate

We **separate the two concerns** and assign each to the appropriate model:

**Identity gate (Stage 3+4) — two-stage local cascade:**

1. *CLIP coarse score.* FLUX generates a single high-quality reference frame
   per shot. CLIP ViT-L/14 computes a normalized mean embedding and compares
   to each character's reference (set on first approved appearance). The
   cosine similarity is checked against `character_threshold = 0.80`.
2. *Qwen2-VL precision check.* Locally-hosted Qwen2-VL (7B operational;
   72B-AWQ downloading) judges identity + attribute + text-image alignment
   in structured form. This catches semantic drifts that pass in CLIP
   embedding space but violate the character description (e.g. "rust-red
   coat" rendered orange) — and conversely rescues CLIP false-negatives.

Both steps run on every shot; the gate passes only when both agree. On
failure, the Cinematographer refines the prompt with the combined feedback
and the shot regenerates (up to 3 retries). The Gemini Vision API is
retained as a drop-in replacement for step 2 but is not on the primary path.

**Motion synthesis (Stage 5)**: The approved reference frame becomes the
*first frame* of a CogVideoX-5b-I2V generation. The shot's text prompt drives
the motion; the image drives the appearance. CogVideoX is trained to maintain
visual identity from the conditioning image throughout the generated clip,
which means our CLIP-approved identity propagates naturally into the motion.

```
        ┌──────────────────────────────┐
        │   FLUX-dev → keyframe        │
        │   "rust-red fox, snowy ..."  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   CLIP gate (cosine sim)     │
        │   pass / refine prompt       │
        └──────────────┬───────────────┘
                pass   │
                       ▼
        ┌──────────────────────────────┐
        │   CogVideoX I2V              │
        │   first_frame + prompt       │
        │     → 6 s video              │
        └──────────────────────────────┘
```

### 6.3 Why this is novel

Pure T2I + interpolation pipelines (e.g. FLUX + RIFE) can preserve identity
*within* a shot but not *between* shots, and produce distorted in-betweens
when the keyframes don't share motion. Pure T2V pipelines (CogVideoX alone,
no I2V conditioning) generate motion but cannot enforce specific visual
identity across shots — each shot's appearance is determined by the text
prompt alone, which is too coarse to fix character details.

**The image-to-video conditioning is the bridge.** We use FLUX's strong text-
to-image quality + a two-stage local identity gate (CLIP coarse + Qwen2-VL
precision) to produce *identity-anchored* reference frames, then use
CogVideoX's I2V capability to extrude them into real motion. The CLIP +
Qwen2-VL cascade is the cross-shot identity enforcer; CogVideoX is the motion
enforcer; together they cover both failure modes — and the entire visual
quality loop runs locally on the workstation GPU.

To my knowledge no published multi-agent system combines these four —
CLIP-similarity coarse gating, local VLM precision gating, automatic prompt
refinement on failure, and image-conditioned video diffusion in a fully local
pipeline.

### 6.4 Reference-embedding update policy

Once set on first appearance, a character's reference embedding is **frozen**.
This prevents drift accumulation: if shot N's embedding became the new
reference, shot N+1's drift would compound. Frozen reference forces every shot
to anchor on the original.

---

## 7. Pipeline Orchestration & Checkpointing

### 7.1 Orchestrator (`src/pipeline/orchestrator.py`)

`PipelineOrchestrator.run(story, duration, style)`:
1. Create `PipelineState.new(...)`.
2. Save initial checkpoint.
3. For each stage in order: time it via `_timed_stage()` context manager,
   run the stage method, save checkpoint on success.
4. Print summary table.
5. Return `state.final_video_path`.

Stages 3+4 are run *jointly per shot* via `_generate_and_check_shot()` to
enable the retry loop. The shot is generated, checked, possibly regenerated,
then approved before moving to the next shot.

### 7.2 Lazy agent loading

```python
AGENT_REGISTRY = {
    "director":         ("src.agents.director",         "DirectorAgent"),
    "cinematographer":  ("src.agents.cinematographer",  "CinematographerAgent"),
    "visual":           ("src.agents.visual",           "VisualAgent"),
    "consistency":      ("src.agents.consistency",      "ConsistencyAgent"),
    "animator":         ("src.agents.animator",         "AnimatorAgent"),
    "narrator":         ("src.agents.narrator",         "NarratorAgent"),
    "post":             ("src.agents.post",             "PostAgent"),
}
```

`python main.py dry-run` executes Director-only without loading FLUX or
CogVideoX.

### 7.3 Checkpoint format

After each stage, `state.model_dump_json(indent=2)` is written to
`checkpoints/<session_id>/<stage_name>.json`. Each file is the full state.
Disk cost: ~50–200 KB per checkpoint. Resume:
```bash
python main.py resume <session_id>
```

### 7.4 Graceful interrupt

SIGINT handler saves an emergency checkpoint, then re-raises.

---

## 8. Tech Stack & Dependencies

### 8.1 Python environment

- Python 3.10 / 3.11 / 3.12 (tested)
- venv recommended over conda (avoids mirror issues)

### 8.2 Core dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `torch` | `≥2.5.1+cu124` (Blackwell needs ≥2.5/cu124, ideally cu126) | Tensor backend |
| `torchvision` | matched to torch | Image transforms |
| `diffusers` | `0.31.0` | FLUX + CogVideoX pipelines |
| `transformers` | `>=4.45,<5.0` | Text encoders |
| `tokenizers` | `>=0.20,<0.21` | Tokenizer backend |
| `accelerate` | `>=0.34.0` | Model offload coordinator |
| `sentencepiece`, `protobuf` | latest | T5 tokenizer requirements |
| `open-clip-torch` | `>=2.24.0` | CLIP ViT-L/14 |
| `openai` | `>=1.40.0` | DeepSeek API client |
| `edge-tts` | `>=6.1.10` | Narration synthesis |
| `moviepy` | `==1.0.3` | Final assembly |
| `opencv-python` | `>=4.10.0` | Frame I/O |
| `pydantic` | `>=2.8.0` | Data models |
| `hf_transfer` | latest | Fast HF downloads (Rust) |
| utilities | latest | pyyaml, pillow, numpy, scipy, tqdm, rich, typer |

### 8.3 External binaries

- **ffmpeg** (apt): used by Edge-TTS speed adjustment + MoviePy encoding.
- **rife-ncnn-vulkan** (optional): only needed if `FORCE_RIFE=1`. Requires
  working Vulkan ICD for the GPU, which many container environments lack.

### 8.4 External services

- **DeepSeek API**: Director, Cinematographer, Narrator. Cost per video:
  ~$0.02–0.03.
- **Gemini Vision API**: optional fallback for the consistency precision
  check; only invoked when local Qwen2-VL is unavailable.
- **Hugging Face Hub** (or `hf-mirror.com` in China): FLUX (gated),
  CogVideoX-5b-I2V (gated), and Qwen2-VL (open) weight downloads.
- **Microsoft Edge-TTS** (no auth, free): narration audio.

### 8.5 Model weights footprint

| Model | Size | Required? |
|---|---:|---|
| FLUX.1-dev | ~32 GB | yes |
| CogVideoX-5b-I2V | ~20 GB | yes |
| CLIP ViT-L/14 (open_clip) | ~900 MB | yes |
| Qwen2-VL-7B-Instruct | ~16 GB | yes (default VLM gate) |
| Qwen2-VL-72B-Instruct-AWQ | ~45 GB | optional heavy-tier (download in progress) |
| RIFE v4.6 (ncnn-vulkan) | ~50 MB | only if FORCE_RIFE=1 |

Total HF cache: ~69 GB with Qwen2-VL-7B; ~114 GB once the 72B-AWQ lands at
`/root/autodl-tmp/hf_cache/qwen2vl-72b-awq`.

---

## 9. Configuration Reference (`config.yaml`)

```yaml
deepseek:
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.7
  max_tokens: 4096
  vision_model: deepseek-chat

# Keyframe synthesizer (FLUX.1-dev)
flux:
  model_id: /root/autodl-tmp/hf_cache/flux   # or HF id: black-forest-labs/FLUX.1-dev
  num_inference_steps: 20
  guidance_scale: 3.5
  width: 1024
  height: 576
  dtype: bfloat16
  enable_cpu_offload: false
  offload_mode: none        # 96 GB GPU; use "model" on 48 GB, "sequential" on 24 GB

# Motion synthesizer (CogVideoX-5b-I2V) — the primary Stage 5 backend
cogvideox:
  model_id: /root/autodl-tmp/hf_cache/cogvideox
  num_inference_steps: 50
  guidance_scale: 6.0
  num_frames: 49            # 49 frames at 8 fps = ~6.1 s per shot
  fps: 8
  width: 720                # native I2V input/output
  height: 480
  dtype: bfloat16
  offload_mode: none        # 96 GB GPU keeps FLUX + CogVideoX both resident

# Legacy interpolator (kept as fallback; opt-in via FORCE_RIFE=1)
rife:
  executable: ./rife-ncnn-vulkan/rife-ncnn-vulkan
  model: rife-v4.6
  interpolation_factor: 4
  gpu_id: 0

tts:
  enabled: true
  voice: en-US-GuyNeural
  speed_clamp: [0.75, 1.25]

video:
  target_fps: 24
  output_codec: libx264
  output_bitrate: 8000k
  width: 1920
  height: 1080

consistency:
  character_threshold: 0.80
  scene_threshold: 0.70
  max_retries_per_shot: 3
  vlm_backend: qwen2vl_7b      # qwen2vl_7b | qwen2vl_72b_awq | gemini | none
  qwen2vl:
    model_id_7b:  /root/autodl-tmp/hf_cache/qwen2vl-7b
    model_id_72b: /root/autodl-tmp/hf_cache/qwen2vl-72b-awq   # ~45 GB, AWQ
    dtype: bfloat16
    offload_mode: none          # 96 GB: keep resident; 48 GB: model offload

pipeline:
  checkpoint_dir: checkpoints
  output_dir: outputs
  max_scenes: 3
  max_shots_per_scene: 2       # at 6 s per CogVideoX clip → 12–36 s output
  log_level: INFO
```

### 9.1 Tuning playbook

| Want | Change |
|---|---|
| Longer video | `max_scenes`, `max_shots_per_scene` up |
| Faster runs (lower quality) | `cogvideox.num_inference_steps: 30` |
| Higher quality | `cogvideox.num_inference_steps: 60`, `flux.num_inference_steps: 25` |
| Stricter character consistency | `consistency.character_threshold: 0.85` |
| Pure offline mode | `tts.enabled: false`, `consistency.vlm_backend: qwen2vl_7b` (default — no API calls in Stage 4) |
| Strongest precision gate | `consistency.vlm_backend: qwen2vl_72b_awq` (after download completes) |
| Cloud fallback only | `consistency.vlm_backend: gemini` (requires `GEMINI_API_KEY`) |
| Lower VRAM (48 GB) | `flux.offload_mode: model`, `cogvideox.offload_mode: model` |
| Lower VRAM (24 GB) | `cogvideox.offload_mode: sequential` — slow but fits |

---

## 10. VRAM & Disk Budget

### 10.1 VRAM (target: NVIDIA RTX PRO 6000, 96 GB, Blackwell)

| Stage active | What's loaded | VRAM |
|---|---|---:|
| Stage 1–2 (LLM only) | nothing on GPU | ~0.3 GB |
| Stage 3 (FLUX resident, no offload) | FLUX-dev transformer + T5 + CLIP-L + VAE | ~24 GB |
| Stage 4a (CLIP coarse) | + open_clip ViT-L/14 | ~26 GB |
| Stage 4b (Qwen2-VL-7B precision) | + Qwen2-VL-7B (bf16) | ~42 GB |
| Stage 4b (Qwen2-VL-72B-AWQ precision) | + Qwen2-VL-72B-AWQ (4-bit) | ~70 GB |
| Stage 5 (FLUX + CogVideoX both resident) | FLUX + CogVideoX-5b transformer + 3D VAE | ~50 GB residency |
| **Stage 5 peak (VAE decode)** | + intermediate latents for 49-frame video | **~60 GB** |
| Stage 7 (CPU-only encoding) | — | ~0.5 GB |

With the 7B VLM in default config, peak (Stage 5 decode) leaves ~35 GB headroom
on 96 GB. With the 72B-AWQ VLM, the orchestrator unloads Qwen2-VL before
entering Stage 5 (the two stages don't overlap), so the 96 GB budget still
fits comfortably.

### 10.2 VRAM (alternative: 48 GB workstation)

| Strategy | Peak VRAM |
|---|---:|
| FLUX `model` offload + CogVideoX `model` offload + FLUX manual offload before Stage 5 | ~28 GB |
| FLUX `sequential` + CogVideoX `model` | ~22 GB |

Set `OFFLOAD_FLUX=1` env var to force FLUX→CPU before Stage 5 in this setup.

### 10.3 VRAM (24 GB minimum: 4090/5080)

Requires `cogvideox.offload_mode: sequential`. Each shot generation takes
~8–10 min. Possible but painful.

### 10.4 Disk

| Component | Size |
|---|---:|
| HF cache (FLUX + CogVideoX + CLIP) | ~53 GB |
| venv (torch + diffusers + etc.) | ~9 GB |
| Per video output (kept) | ~150 MB |
| Working files during run | ~1 GB |
| **Total** | **~63 GB minimum** |

Comfortable on an 80–100 GB data disk; tight on 50 GB.

---

## 11. Runtime Performance & Timing

### 11.1 Per-stage budget on 96 GB PRO 6000 (no offload)

For a 2-scene, 4-shot run (~24 s output video):

| Stage | Wall time | Bottleneck |
|---|---:|---|
| 1. Director | ~20 s | DeepSeek API latency |
| 2. Cinematographer | ~30 s | 4 sequential DeepSeek calls |
| 3. Visual (FLUX, 4 keyframes × ~5 s) | ~25 s | GPU compute |
| 4. Consistency (batch mode) | ~17 s + 3.6 s/shot | CLIP encode (sub-second) + one Qwen-72B-AWQ load + per-shot inference. For 4 shots: ~31 s total vs. ~82 s in per-shot mode. |
| **5. Animator (CogVideoX, 4 shots × ~3 min)** | **~12 min** | GPU compute, dominant cost |
| 6. Narrator | ~1 min | Edge-TTS + ffmpeg |
| 7. Post | ~30 s | MoviePy + libx264 |
| **Total (warm cache)** | **~15 min** | Stage 5 dominates |
| **Total (cold, includes one-time weight download)** | +30–60 min | network speed |

### 11.2 Where time goes

```
Stage 5 (CogVideoX): ███████████████████████████████████████████████████████  75-85%
Stage 3 (FLUX):      ████  3-5%
Stage 6 (TTS):       ███  3-4%
Stage 4 (Consistency): ███  3-4%
Stage 7 (Post):      ██  2%
Stages 1+2 (LLM):    █  1-2%
```

### 11.3 Cost (DeepSeek API only)

~$0.02–0.03 per video. DeepSeek pricing makes this near-trivial.

---

## 12. File & Directory Layout

```
storyforge/
├── README.md                       # public-facing intro
├── PROJECT.md                      # this document
├── SETUP_PRO6000.md                # remote setup runbook
├── requirements.txt
├── config.yaml
├── .env.example
├── .env                            # secrets (gitignored)
├── .gitignore
├── main.py                         # Typer CLI entry point
│
├── docs/
│   ├── ARCHITECTURE.md             # superseded by PROJECT.md §3
│   ├── PIPELINE.md                 # superseded by §5
│   ├── AGENTS.md                   # superseded by §5
│   ├── TECH_STACK.md               # superseded by §8
│   └── REPORT_OUTLINE.md           # paper structure (kept; orthogonal)
│
├── src/
│   ├── schemas/data_models.py      # Pydantic models
│   ├── utils/
│   │   ├── llm_client.py           # DeepSeek wrapper
│   │   ├── prompt_templates.py     # agent prompt templates
│   │   └── video_utils.py          # ffmpeg/OpenCV helpers
│   ├── models/
│   │   ├── flux_wrapper.py         # FLUX/SDXL auto-detecting wrapper
│   │   ├── clip_wrapper.py         # CLIP ViT-L/14 wrapper
│   │   ├── cogvideox_wrapper.py    # CogVideoX-5b-I2V wrapper (new)
│   │   └── rife_wrapper.py         # legacy RIFE subprocess driver
│   ├── pipeline/
│   │   ├── orchestrator.py         # state machine + retry loop
│   │   └── checkpoint.py           # save/load/list/latest
│   └── agents/
│       ├── director.py
│       ├── cinematographer.py
│       ├── visual.py
│       ├── consistency.py
│       ├── animator.py             # now uses CogVideoX by default
│       ├── narrator.py
│       └── post.py
│
├── rife-ncnn-vulkan/               # optional; only if FORCE_RIFE=1
├── outputs/<session_id>/           # gitignored
│   ├── keyframes/<shot_id>/        # FLUX reference frames
│   ├── clips/                      # CogVideoX per-shot MP4s
│   ├── audio/                      # narration WAVs
│   └── final_video.mp4
└── checkpoints/<session_id>/       # gitignored
```

---

## 13. CLI Usage

```bash
# Full generation
python main.py run "STORY..." [--duration N] [--style "..."] [--config path]

# Director-only (no GPU; cheap; sanity-check API and prompts)
python main.py dry-run "STORY..." [--duration N] [--style "..."]

# Resume from last checkpoint
python main.py resume <session_id> [--config path]

# List sessions
python main.py list-sessions [--config path]
```

### Environment flags

| Var | Effect |
|---|---|
| `FORCE_RIFE=1` | Use RIFE interpolation instead of CogVideoX (needs Vulkan) |
| `STATIC_ONLY=1` | Render each shot as a static still (debug fallback) |
| `OFFLOAD_FLUX=1` | Force FLUX→CPU before Stage 5 (low-VRAM machines) |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | Use Rust-based parallel downloads |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce fragmentation |

---

## 14. Deployment

See **`SETUP_PRO6000.md`** for the complete runbook. Summary:

```bash
# LOCAL (Windows Git Bash):
rsync -avzP -e "ssh -p <PORT>" --exclude='outputs/' --exclude='checkpoints/' \
    --exclude='__pycache__/' --exclude='.venv/' --exclude='*.pyc' \
    /d/CS/Graphics/PJ3/ root@<HOST>:/root/autodl-tmp/storyforge/

# REMOTE:
ssh -p <PORT> root@<HOST>
source /etc/network_turbo

# Fresh venv on data disk
python3 -m venv /root/autodl-tmp/venvs/sf
source /root/autodl-tmp/venvs/sf/bin/activate
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch for Blackwell
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

cd /root/autodl-tmp/storyforge
pip install -r requirements.txt
pip uninstall -y flash-attn flash_attn kernels 2>/dev/null
pip install hf_transfer

apt update && apt install -y ffmpeg unzip libvulkan1

# HF setup — set HF_HOME ONLY, let HF_HUB_CACHE default
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login

# Download both models (accept licenses in browser first)
hf download black-forest-labs/FLUX.1-dev
hf download THUDM/CogVideoX-5b-I2V

# Verify and run
python -c "from src.schemas.data_models import PipelineConfig; \
    print(PipelineConfig.from_yaml('config.yaml').cogvideox.model_id)"

set -a && source .env && set +a
tmux new -s run
python main.py run "A red fox stands in falling snow." --duration 5 \
    --style "cinematic, photorealistic"
```

---

## 15. Failure Modes & Recovery

| Symptom | Cause | Fix |
|---|---|---|
| `ValidationError` in Director | Malformed JSON from LLM | Auto-retried 3× with error feedback |
| Stage 3 OOM | FLUX residency too aggressive | Set `flux.offload_mode: model` (48 GB) or `sequential` (24 GB) |
| Stage 5 OOM | CogVideoX residency too aggressive | Set `cogvideox.offload_mode: model` or `sequential`; set `OFFLOAD_FLUX=1` |
| CogVideoX import error | diffusers version mismatch | `pip install --upgrade diffusers` (need ≥0.30 for CogVideoX) |
| FluxPipeline import error w/ FA3 backtrace | flash-attn / kernels installed | `pip uninstall -y flash-attn flash_attn kernels` |
| CUDA: False on Blackwell GPU | PyTorch too old (cu121) | Install PyTorch ≥ 2.5 with cu124 or cu126 |
| HF downloads slow (<1 MB/s) | Mirror congestion | `pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1` |
| HF re-downloads despite cache | `HF_HUB_CACHE` set to wrong path | `unset HF_HUB_CACHE`, only export `HF_HOME` |
| RIFE produces 4 frames per shot | Vulkan ICD missing in container | Use CogVideoX (default); only set `FORCE_RIFE=1` where Vulkan works |
| Crash mid-run | various | `python main.py list-sessions; python main.py resume <id>` |

---

## 16. Evaluation Plan

### 16.1 Test stories

| ID | Prompt | Duration | Stresses |
|---|---|---|---|
| S1 | "A red fox crosses a snowy forest at dusk." | 12 s | Single character, motion coherence |
| S2 | "An astronaut on Mars discovers a metallic sphere." | 18 s | Rare object, prop consistency |
| S3 | "Two friends meet at a Tokyo café at night." | 18 s | Two characters, dialogue framing |
| S4 | "A dragon flies over mountains at dawn." | 12 s | Action shots, fast motion |
| S5 | "A seed grows into a tree, then forest." | 24 s | Scale + time progression |

### 16.2 Metrics

| Metric | Definition | Computed via |
|---|---|---|
| **CLIP-T** | Per-frame text-image similarity vs shot prompt | CLIP-L/14 |
| **CLIP-I** | Inter-shot character consistency vs reference embedding | CLIP-L/14 |
| **MOTION** | Inter-frame optical-flow magnitude (mean across video) | OpenCV `calcOpticalFlowFarneback` |
| **TIME** | End-to-end generation wall time | Pipeline logs |
| **RETRY** | Avg consistency retry count per shot | Pipeline logs |
| **SUBJECTIVE** | Author-graded 1–5 on coherence, faithfulness, motion plausibility | Manual |

### 16.3 Ablations

| Ablation | What's varied | Hypothesis |
|---|---|---|
| **A1**: No consistency gate | Skip Stage 4 | CLIP-I drops ~0.10; visible character drift |
| **A2**: Gate without refinement | Stage 4 fails → skip shot | Intermediate CLIP-I; some shots missing |
| **A3**: CogVideoX without FLUX seed | Use text-only prompt (T2V) | CLIP-I drops; identity drifts each shot |
| **A4**: FLUX + RIFE (no CogVideoX) | Legacy path | MOTION higher but spatial distortion |

### 16.4 Baseline

Direct CogVideoX-5b-T2V (text-to-video, no image conditioning). Compare CLIP-I
and qualitative coherence vs StoryForge's image-conditioned approach.

---

## 17. Known Limitations & Future Work

### 17.1 Honest limitations

1. **CogVideoX is 6-second-bound**. Longer per-shot videos require either
   another model or repeated I2V conditioning across multi-second segments.
2. **CLIP-similarity approximates identity**. A dedicated identity encoder
   (ArcFace, IP-Adapter) would be more discriminative.
3. **Generation time is dominated by CogVideoX**. ~3 min/shot on a 96 GB GPU;
   not real-time.
4. **No physical plausibility constraints**. CogVideoX can hallucinate
   impossible geometry on long-tail prompts.
5. **Vulkan-dependent fallback path is fragile**. RIFE is preserved but most
   container deployments cannot use it.

### 17.2 Future work

1. **Replace CLIP gate with IP-Adapter** for explicit identity injection at
   the FLUX stage. Would tighten cross-shot consistency further.
2. **Shot-to-shot motion continuity**: feed the *last frame* of shot N as the
   *first frame* of shot N+1 (when characters are continuous), removing the
   visible cut between shots that share a subject.
3. **User-in-the-loop editing**: pause between stages for prompt overrides.
4. **Beam search over shot variants**: generate K candidates per shot in
   parallel; select highest CLIP-I.
5. **Audio scape**: generate ambient SFX via MusicGen / AudioGen, mix in
   Stage 7.

---

## 18. Architectural Evolution: Why CogVideoX Replaced RIFE

The original design used FLUX → many keyframes → RIFE interpolation. The
revised design uses FLUX → one keyframe → CogVideoX I2V → real motion video.

### 18.1 What we learned about the original design

The RIFE-based pipeline rested on a critical assumption: that consecutive
FLUX-generated keyframes share enough motion structure for optical flow to
bridge them. **This assumption is false.** Each FLUX keyframe is generated
from an independent denoising trajectory; even with the same prompt and
similar seeds, the resulting images differ in pose, composition, and fine
detail in ways that classical optical flow cannot reconcile. Empirically,
attempting `ffmpeg minterpolate` between such frames produced severe spatial
distortion (subjects melted between poses).

Even with working RIFE (neural interpolation), the same fundamental issue
applies: RIFE was trained on consecutive video frames, where the inter-frame
delta is small and motion-coherent. FLUX keyframes are nothing like that.

### 18.2 What CogVideoX gives us

CogVideoX-5b-I2V is a video diffusion model. It does not interpolate between
given frames — it *synthesizes* motion conditioned on a starting image and a
text prompt. The motion is generated *de novo* in a way that's globally
coherent for the entire clip duration.

This re-purposes the FLUX keyframe from "one of many frames to interpolate
between" to "a strong visual prior that anchors the appearance of the
generated motion clip." The CLIP consistency gate now operates on a single
reference frame per shot, not a sequence — simpler and stricter.

### 18.3 What the report says

The report acknowledges this evolution explicitly in Section 4 (Architecture):

> "An earlier iteration of StoryForge used frame interpolation (RIFE) to
> bridge multiple FLUX-generated keyframes into a continuous shot. We
> determined that this approach is fundamentally limited because
> independently-generated diffusion frames lack the motion coherence that
> classical or neural flow estimators require. Switching to image-conditioned
> video diffusion (CogVideoX-5b-I2V) addresses both the within-shot motion
> problem and the cross-shot identity problem with a single substitution."

### 18.4 The RIFE path is preserved as a fallback

For deployments where Vulkan is configured and CogVideoX is not available,
the original RIFE-based animator is still callable via `FORCE_RIFE=1`. This
also enables the report's ablation study (A4: FLUX+RIFE baseline) on
identical input prompts.

---

## 19. Glossary

- **Keyframe (current design)**: A single FLUX-generated reference image per
  shot, used as the first frame of CogVideoX I2V.
- **Shot**: A continuous segment of video from one camera setup. Typically
  6.125 s (CogVideoX native).
- **Scene**: A coherent narrative unit. Contains multiple shots.
- **Consistency gate**: The two-stage check between Stage 3 (Visual) and
  Stage 5 (Animator) — CLIP ViT-L/14 for coarse identity matching, then
  Qwen2-VL for precision text-image and attribute judgment — that decides
  whether a shot's keyframe is good enough.
- **Qwen2-VL**: Local open-weights vision-language model used as the
  precision-tier consistency check. 7B variant is integrated; 72B-AWQ
  (4-bit quantized, ~45 GB) is the heavy-tier check, downloading at the
  time of this revision.
- **Reference embedding**: A character's signature CLIP vector, set on first
  approval, frozen thereafter.
- **CogVideoX-I2V**: Image-to-Video variant of CogVideoX-5b. Takes a first
  frame + text prompt; produces a coherent motion video.
- **CPU offload**: Streaming model weights between RAM and VRAM during
  inference. `model` = swap whole submodules; `sequential` = swap layer-by-layer.
- **bf16 / bfloat16**: 16-bit float with fp32 exponent range.
- **DeepSeek API**: OpenAI-compatible LLM API used for all language reasoning.
- **AutoDL**: Chinese cloud GPU rental service used in development.

---

## Appendix — useful one-liners

```bash
# Live VRAM monitoring
watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader'

# Outputs directory size
du -sh outputs/* checkpoints/* /root/autodl-tmp/hf_cache 2>/dev/null

# Tail the latest log
tail -f outputs/$(ls -t outputs | head -1)/run.log

# Extract a still frame from final video for the report
ffmpeg -i final_video.mp4 -ss 00:00:05 -vframes 1 frame_5s.png

# Compare two keyframes via CLIP
python -c "
from src.models.clip_wrapper import get_clip
c = get_clip()
import sys
print('sim:', c.cosine_similarity(c.encode_image(sys.argv[1]), c.encode_image(sys.argv[2])))
" outputs/.../keyframes/scene_1_sh1/kf001.png outputs/.../keyframes/scene_2_sh1/kf001.png
```

**End of PROJECT.md.** Update when: architecture changes, config schema
changes, new failure mode found, or eval results come in.
