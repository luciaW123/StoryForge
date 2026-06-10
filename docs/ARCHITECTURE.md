# Architecture: StoryForge

## 1. System Overview

StoryForge follows a **linear-with-feedback** agent architecture. Agents execute in a fixed order (pipeline), but the Consistency Agent introduces a local feedback loop that can trigger regeneration of individual shots before the pipeline advances.

```
                    ┌─────────────────────────────────────┐
                    │          PipelineOrchestrator        │
                    │  (state machine + retry logic)       │
                    └──────────────┬──────────────────────┘
                                   │ manages
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   [Text Agents]           [Vision Agents]          [Media Agents]
   DeepSeek API            FLUX.1 + CLIP            RIFE + MoviePy
   Director                Visual                   Animator
   Cinematographer         Consistency              Narrator
                                                    Post
```

---

## 2. Full Pipeline Diagram

```
USER INPUT: story text (str), duration (int), style (str)
    │
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 1 — Director Agent                           ║
║  Model: DeepSeek-chat (API)                         ║
║  Input:  raw story string                           ║
║  Output: ScriptPlan (JSON)                          ║
║   • scene list (3-8 scenes)                         ║
║   • character roster + visual descriptions          ║
║   • global art style + mood                         ║
║   • shot list per scene (1-3 shots each)            ║
╚══════════════════════════════════════════════════════╝
    │  ScriptPlan
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 2 — Cinematographer Agent                    ║
║  Model: DeepSeek-chat (API)                         ║
║  Input:  ScriptPlan                                 ║
║  Output: ShotPlan (JSON, extends ScriptPlan)        ║
║   • FLUX prompt per shot (detailed, ~80 tokens)     ║
║   • negative prompt per shot                        ║
║   • camera angle / movement descriptor              ║
║   • transition type (cut / fade / dissolve)         ║
║   • duration_sec per shot                           ║
║   • num_keyframes per shot                          ║
╚══════════════════════════════════════════════════════╝
    │  ShotPlan
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 3 — Visual Agent         ┐                   ║
║  Model: FLUX.1-dev (local 4090) │ per-shot loop     ║
║  Input:  Shot (flux_prompt,     │                   ║
║          num_keyframes, seed)   │                   ║
║  Output: keyframe PNG images    │                   ║
║   • N keyframes per shot        │                   ║
║   • 1024×576 or 1024×1024       │                   ║
║   • consistent seed per char    │                   ║
╚══════════════════════════════════╪═══════════════════╝
    │  keyframe images             │
    ▼                             │ if score < threshold
╔══════════════════════════════════╪═══════════════════╗
║  STAGE 4 — Consistency Agent    │                   ║
║  Model: CLIP-L/14 + DeepSeek   │                   ║
║  Input:  keyframes, ShotPlan   │                   ║
║  Output: consistency_score,    │                   ║
║          feedback (str)        │                   ║
║   • CLIP cosine similarity     │                   ║
║     between shots sharing      ├───────────────────┘
║     same character/scene       │ (max 3 retries)
║   • DeepSeek critique of       │
║     visual-text alignment      │
║   • per-shot PASS / FAIL       │
╚════════════════════════════════╝
    │  approved keyframes (all shots)
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 5 — Animator Agent                           ║
║  Model: RIFE v4.6 (local 4090)                      ║
║  Input:  keyframe PNGs per shot                     ║
║  Output: video clip MP4 per shot                    ║
║   • 2x or 4x frame interpolation                    ║
║   • target 24 fps                                   ║
║   • per-shot .mp4 (silent)                          ║
╚══════════════════════════════════════════════════════╝
    │  per-shot video clips
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 6 — Narrator Agent  (optional flag)          ║
║  Model: DeepSeek + Edge-TTS                         ║
║  Input:  ShotPlan, per-shot duration                ║
║  Output: per-shot WAV audio                         ║
║   • DeepSeek writes narration script                ║
║   • Edge-TTS synthesizes audio                      ║
║   • speed-adjusted to match video duration          ║
╚══════════════════════════════════════════════════════╝
    │  video clips + audio files
    ▼
╔══════════════════════════════════════════════════════╗
║  STAGE 7 — Post Agent                               ║
║  Library: MoviePy                                   ║
║  Input:  all shot clips, audio, ShotPlan            ║
║  Output: final MP4 (1080p or 720p)                  ║
║   • apply transitions (fade/dissolve/cut)           ║
║   • overlay audio tracks                            ║
║   • optional: subtitle burn-in                      ║
║   • optional: background music (user-provided)      ║
║   • export final video                              ║
╚══════════════════════════════════════════════════════╝
    │
    ▼
OUTPUT: final_video.mp4
```

---

## 3. State Machine

The Orchestrator is a state machine. States:

```
INIT
 │
 ▼
SCRIPTING      ← Director Agent
 │
 ▼
PLANNING       ← Cinematographer Agent
 │
 ▼
GENERATING     ← Visual Agent (per shot)
 │  ↑
 ▼  │ retry (max 3x per shot)
CHECKING       ← Consistency Agent
 │
 ▼
INTERPOLATING  ← Animator Agent
 │
 ▼
NARRATING      ← Narrator Agent (optional)
 │
 ▼
EDITING        ← Post Agent
 │
 ▼
DONE
```

Checkpoints are saved to disk at each state transition. On crash/restart, the pipeline resumes from the last completed checkpoint.

---

## 4. Data Models

All models defined in `src/schemas/data_models.py` using Pydantic v2.

```python
class Character(BaseModel):
    id: str                              # e.g. "char_fox"
    name: str                            # e.g. "The Fox"
    description: str                     # natural language
    visual_prompt: str                   # FLUX-optimized visual description
    reference_seed: int                  # fixed seed for consistent appearance
    clip_embedding: Optional[list[float]] = None  # set after first generation

class Shot(BaseModel):
    id: str                              # e.g. "s1_sh2"
    scene_id: str
    order: int                           # global ordering
    description: str                     # natural language shot description
    flux_prompt: str                     # full FLUX positive prompt
    negative_prompt: str
    camera_angle: str                    # "low angle", "bird's eye", "eye level" ...
    camera_movement: str                 # "static", "slow pan left", "zoom in" ...
    transition_out: str                  # "cut", "fade_to_black", "dissolve"
    duration_sec: float
    num_keyframes: int
    characters_present: list[str]        # list of character IDs
    # filled during execution:
    keyframe_paths: list[str] = []
    video_clip_path: Optional[str] = None
    audio_clip_path: Optional[str] = None
    consistency_score: float = 0.0
    approved: bool = False
    retry_count: int = 0

class Scene(BaseModel):
    id: str                              # e.g. "scene_1"
    order: int
    title: str
    setting: str
    time_of_day: str                     # "dawn", "noon", "dusk", "night"
    mood: str                            # "tense", "peaceful", "melancholic" ...
    description: str
    shots: list[Shot]

class ScriptPlan(BaseModel):
    title: str
    genre: str
    overall_mood: str
    art_style: str                       # e.g. "cinematic, photorealistic, 35mm film"
    global_style_suffix: str            # appended to every FLUX prompt
    color_palette: str                   # e.g. "muted blues and warm oranges"
    characters: list[Character]
    scenes: list[Scene]
    total_duration_sec: float

class PipelineState(BaseModel):
    session_id: str
    stage: str
    story_input: str
    script_plan: Optional[ScriptPlan] = None
    output_dir: str
    final_video_path: Optional[str] = None
    created_at: str
    total_shots: int = 0
    approved_shots: int = 0
```

---

## 5. Agent Communication Protocol

Agents do **not** call each other directly. All communication goes through `PipelineState`:

```
Agent.process(state: PipelineState) -> PipelineState
```

Each agent reads what it needs from state, does its work, writes results back, returns updated state. The Orchestrator passes state between agents sequentially.

This makes each agent independently testable and the pipeline resumable at any stage.

---

## 6. Consistency Mechanism (Key Innovation)

The Consistency Agent addresses the core challenge of multi-shot visual coherence:

```
For each shot containing character C:
  1. Extract CLIP-L/14 embedding of character region (or whole frame)
  2. Compare cosine similarity to reference embedding of C
     (reference = first approved shot featuring C)
  3. If similarity < 0.82 → FAIL → trigger prompt refinement + regeneration

For scene continuity:
  1. Compare background CLIP embeddings of consecutive shots in same scene
  2. Dramatic shift in bg embedding → flag for human review (warning, not block)
```

Character reference embeddings are stored in `Character.clip_embedding` after the first approved generation.

---

## 7. Keyframe Count Formula

Given a shot of `duration_sec` seconds, target output at 24fps with RIFE 4x interpolation:

```
num_keyframes = ceil(duration_sec * 24 / 4) + 1
             = ceil(duration_sec * 6) + 1

Example: 3-second shot
  keyframes = ceil(18) + 1 = 19 keyframes
  after 4x RIFE: 19 * 4 = 76 frames ≈ 3.2 seconds at 24fps  ✓
```

For RIFE 2x (faster, slightly lower quality):
```
num_keyframes = ceil(duration_sec * 12) + 1
```

---

## 8. Hardware Utilization Map

```
CPU (any):
  - All DeepSeek API calls (async)
  - MoviePy editing
  - Edge-TTS synthesis
  - State management + orchestration

RTX 4090 (24GB VRAM):
  - FLUX.1-dev inference       ~14-16GB VRAM, ~8-15s per image @ 20 steps
  - CLIP-L/14 embeddings       ~1GB VRAM
  - RIFE frame interpolation   ~2-4GB VRAM, ~0.5s per frame pair

Total estimated time per minute of output video:
  - FLUX generation:    ~15 shots × 4 keyframes × 10s = ~600s = 10 min
  - RIFE interpolation: ~15 shots × 72 frames × 0.5s = ~540s = 9 min
  - DeepSeek + post:    ~2-3 min (API latency + MoviePy)
  - TOTAL:              ~20-25 min per 1 minute of video
```
