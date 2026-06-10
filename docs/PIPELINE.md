# Pipeline Specification: StoryForge

Detailed input/output contracts and logic for each stage.

---

## Stage 1: Director Agent

**File**: `src/agents/director.py`  
**Purpose**: Parse raw story text into a structured, machine-readable script plan.

### Input

```python
story_input: str          # Raw user story (any length)
target_duration_sec: int  # Desired total video length (e.g. 20)
style_hint: str           # e.g. "cinematic, photorealistic"
```

### LLM Call

**System prompt** (from `prompt_templates.py`):
```
You are a film director specializing in short-form animated storytelling.
Your job is to break down a story into a precise, structured screenplay.
Always respond in valid JSON matching the ScriptPlan schema.
```

**User prompt**:
```
Story: {story_input}
Target duration: {target_duration_sec} seconds
Style: {style_hint}

Break this story into 3-8 scenes with 1-3 shots each.
For each character, write a concise visual description suitable for image generation.
Choose a cohesive art style and color palette that fits the mood.
Distribute scene durations to sum to approximately {target_duration_sec} seconds.
Respond ONLY with valid JSON matching the ScriptPlan schema.
```

### Output: `ScriptPlan` JSON

```json
{
  "title": "The Fox and the Moon",
  "genre": "nature documentary",
  "overall_mood": "serene, contemplative",
  "art_style": "cinematic photorealistic, 35mm film grain, shallow depth of field",
  "global_style_suffix": ", cinematic lighting, 35mm film, photorealistic, 8k",
  "color_palette": "cool blues and warm amber highlights",
  "characters": [
    {
      "id": "char_fox",
      "name": "The Fox",
      "description": "A red fox with amber fur and a white-tipped tail",
      "visual_prompt": "a red fox with vibrant amber fur, white-tipped bushy tail, bright amber eyes, graceful posture",
      "reference_seed": 42
    }
  ],
  "scenes": [
    {
      "id": "scene_1",
      "order": 1,
      "title": "Forest Entrance",
      "setting": "snow-covered pine forest edge",
      "time_of_day": "dusk",
      "mood": "quiet, anticipatory",
      "description": "The fox emerges from dense pines into a moonlit clearing.",
      "shots": [
        {
          "id": "s1_sh1",
          "scene_id": "scene_1",
          "order": 1,
          "description": "Wide shot: fox steps out of tree line, snow falling gently",
          "camera_angle": "eye level",
          "camera_movement": "slow push in",
          "transition_out": "dissolve",
          "duration_sec": 4.0,
          "characters_present": ["char_fox"]
        }
      ]
    }
  ],
  "total_duration_sec": 20.0
}
```

### Error Handling

- If LLM returns malformed JSON → retry up to 3x with stricter prompt
- If scenes sum to wrong duration → rescale proportionally
- If > 8 scenes → ask LLM to consolidate (too many shots = slow generation)

---

## Stage 2: Cinematographer Agent

**File**: `src/agents/cinematographer.py`  
**Purpose**: Transform the abstract shot list into concrete, generation-ready FLUX prompts and timing specs.

### Input

```python
state: PipelineState  # Contains script_plan with shots (no flux_prompt yet)
```

### LLM Call (per-shot, batched)

**System prompt**:
```
You are a cinematographer and prompt engineer for AI image generation using FLUX.1.
Write detailed, FLUX-optimized image generation prompts.
Prompts should be ~60-100 tokens, describing the visual frame precisely.
Include: subject action, environment details, lighting, atmosphere, camera description.
Do NOT include style tokens (those are added automatically).
Always respond in valid JSON.
```

**User prompt** (for each shot):
```
Shot: {shot.description}
Scene setting: {scene.setting}, {scene.time_of_day}
Mood: {scene.mood}
Characters in shot: {[char.visual_prompt for char in characters_present]}
Camera: {shot.camera_angle}, {shot.camera_movement}
Art style context: {script_plan.art_style}
Color palette: {script_plan.color_palette}

Write:
1. flux_prompt: A detailed positive prompt for this single frame
2. negative_prompt: What to avoid
3. num_keyframes: How many keyframes to generate (use formula: ceil(duration*6)+1)
   Shot duration: {shot.duration_sec}s

Respond ONLY with JSON: {"flux_prompt": "...", "negative_prompt": "...", "num_keyframes": N}
```

### Output (fills in Shot fields)

```python
shot.flux_prompt = "A red fox with vibrant amber fur stepping through deep snow at the edge of a pine forest, golden hour dusk light filtering through trees, soft snowfall, breath visible in cold air, foreground snow bokeh, eye-level perspective, push-in motion"
shot.negative_prompt = "cartoon, anime, painting, blurry, low quality, watermark, text, multiple foxes, humans"
shot.num_keyframes = 25  # ceil(4.0 * 6) + 1
```

The `global_style_suffix` is appended to every `flux_prompt` automatically by the orchestrator:
```python
final_prompt = shot.flux_prompt + script_plan.global_style_suffix
# → "...push-in motion, cinematic lighting, 35mm film, photorealistic, 8k"
```

### Prompt Refinement (called during retry loop)

If Consistency Agent returns FAIL with feedback, Cinematographer refines:

```python
def refine_prompt(self, shot: Shot, feedback: str) -> str:
    # LLM call: given original prompt + failure feedback → improved prompt
```

**Refinement prompt**:
```
Original FLUX prompt: {shot.flux_prompt}
Consistency feedback: {feedback}
Retry count: {shot.retry_count}/3

Rewrite the prompt to address the consistency issue while preserving the shot intent.
Respond with ONLY the new flux_prompt string.
```

---

## Stage 3: Visual Agent

**File**: `src/agents/visual.py` + `src/models/flux_wrapper.py`  
**Purpose**: Generate keyframe images for each shot using FLUX.1-dev.

### Input (per shot)

```python
shot.flux_prompt: str
shot.negative_prompt: str  
shot.num_keyframes: int
shot.reference_seed: int   # from character's reference_seed if solo shot; else random
script_plan.global_style_suffix: str
```

### FLUX.1-dev Configuration

```python
# src/models/flux_wrapper.py

from diffusers import FluxPipeline
import torch

class FluxWrapper:
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()  # save VRAM for CLIP concurrency

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        num_images: int,
        seed: int,
        width: int = 1024,
        height: int = 576,   # 16:9
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
    ) -> list[PIL.Image]:
        generator = torch.Generator("cuda").manual_seed(seed)
        results = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return results.images
```

### Keyframe Generation Strategy

For a shot with N keyframes:
- **Single character, static shot**: use same seed for all N frames (near-identical, subtle variation)
- **Movement shot** (pan/zoom): vary seed slightly (+1 per frame) to allow natural motion
- **Scene transition**: use distinct seed, ensure consistent background elements via prompt

```python
def generate_keyframes(self, shot: Shot, output_dir: Path) -> list[Path]:
    paths = []
    for i in range(shot.num_keyframes):
        seed = shot.reference_seed + i  # slight variation for motion
        img = self.flux.generate(
            prompt=shot.flux_prompt,
            negative_prompt=shot.negative_prompt,
            num_images=1,
            seed=seed,
        )[0]
        path = output_dir / f"{shot.id}_kf{i:03d}.png"
        img.save(path)
        paths.append(path)
    return paths
```

### Output

```
outputs/{session_id}/keyframes/
    s1_sh1_kf000.png
    s1_sh1_kf001.png
    ...
    s1_sh1_kf024.png
```

---

## Stage 4: Consistency Agent

**File**: `src/agents/consistency.py` + `src/models/clip_wrapper.py`  
**Purpose**: Score visual coherence. Gate keyframes before interpolation. Trigger retry if needed.

### Two-Level Consistency Check

**Level 1 — Character Identity Consistency**

For each shot containing character C:

```python
def check_character_consistency(self, shot: Shot, character: Character) -> tuple[float, str]:
    # 1. Extract CLIP embedding from each keyframe
    embeddings = [self.clip.encode(img) for img in shot.keyframes]
    
    # 2. If character has no reference yet, set first embedding as reference
    if character.clip_embedding is None:
        character.clip_embedding = embeddings[0]
        return 1.0, "Reference set"
    
    # 3. Compute cosine similarity to reference
    ref = np.array(character.clip_embedding)
    scores = [cosine_sim(ref, e) for e in embeddings]
    mean_score = np.mean(scores)
    
    # 4. Pass/fail
    if mean_score < CHARACTER_CONSISTENCY_THRESHOLD:  # 0.80
        feedback = f"Character visual identity inconsistent (score={mean_score:.2f}). "
                   f"Ensure '{character.visual_prompt}' is prominent in frame."
        return mean_score, feedback
    return mean_score, "OK"
```

**Level 2 — Scene Continuity (advisory)**

For consecutive shots in the same scene:

```python
def check_scene_continuity(self, shot_a: Shot, shot_b: Shot) -> float:
    # Compare background CLIP embeddings
    bg_score = cosine_sim(shot_a.scene_embedding, shot_b.scene_embedding)
    if bg_score < SCENE_CONTINUITY_THRESHOLD:  # 0.70
        logger.warning(f"Scene continuity warning between {shot_a.id} and {shot_b.id}: {bg_score:.2f}")
    return bg_score
```

**Level 3 — Text-Image Alignment (DeepSeek)**

```python
def check_text_alignment(self, shot: Shot) -> tuple[bool, str]:
    # Encode first keyframe as base64, send to DeepSeek vision
    image_b64 = encode_image(shot.keyframe_paths[0])
    prompt = f"""
    Does this image match the description: "{shot.description}"?
    Art style should be: "{script_plan.art_style}"
    
    Answer YES or NO, then one sentence of feedback.
    """
    response = self.llm.chat(prompt, image=image_b64)
    passed = response.startswith("YES")
    return passed, response
```

### Decision Logic

```python
def evaluate_shot(self, shot: Shot) -> tuple[bool, str]:
    # All characters must pass Level 1
    for char_id in shot.characters_present:
        char = self.get_character(char_id)
        score, feedback = self.check_character_consistency(shot, char)
        shot.consistency_score = score
        if score < CHARACTER_CONSISTENCY_THRESHOLD:
            return False, feedback
    
    # Level 3 text alignment
    passed, feedback = self.check_text_alignment(shot)
    if not passed:
        return False, feedback
    
    shot.approved = True
    return True, "OK"
```

### Thresholds (configurable in `config.yaml`)

```yaml
consistency:
  character_threshold: 0.80    # CLIP cosine similarity
  scene_threshold: 0.70        # background continuity
  max_retries_per_shot: 3
  use_deepseek_vision: true    # set false to skip LLM check (faster)
```

---

## Stage 5: Animator Agent

**File**: `src/agents/animator.py` + `src/models/rife_wrapper.py`  
**Purpose**: Interpolate keyframes into smooth video clips.

### Input (per shot)

```python
shot.keyframe_paths: list[Path]   # e.g. 25 PNG files
shot.duration_sec: float          # target duration
target_fps: int                   # 24
interpolation_factor: int         # 2 or 4 (from config)
```

### RIFE Interpolation

```python
# src/models/rife_wrapper.py

import subprocess
from pathlib import Path

class RIFEWrapper:
    def __init__(self, rife_executable: str, model: str = "rife-v4.6"):
        self.exe = rife_executable  # path to rife-ncnn-vulkan binary
        self.model = model
    
    def interpolate(
        self,
        input_frames_dir: Path,
        output_frames_dir: Path,
        factor: int = 4,   # 2x or 4x
    ) -> None:
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            self.exe,
            "-i", str(input_frames_dir),
            "-o", str(output_frames_dir),
            "-m", self.model,
            "-n", str(factor),
            "-g", "0",   # GPU id
        ], check=True)
```

### Assembly into MP4

After interpolation, frames are assembled into a video clip:

```python
def frames_to_clip(frames_dir: Path, output_path: Path, fps: int = 24) -> Path:
    import cv2
    frames = sorted(frames_dir.glob("*.png"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()
    return output_path
```

### Output

```
outputs/{session_id}/clips/
    s1_sh1.mp4    # 4.0s @ 24fps = 96 frames
    s1_sh2.mp4
    ...
```

---

## Stage 6: Narrator Agent

**File**: `src/agents/narrator.py`  
**Purpose**: Generate and synthesize voiceover audio synchronized to video clips.

### Input (per shot)

```python
shot.description: str
shot.duration_sec: float
script_plan.overall_mood: str
```

### Step 1: Script Generation (DeepSeek)

```
Write a brief narrator voiceover for this shot.
Shot: {shot.description}
Mood: {script_plan.overall_mood}
Duration: {shot.duration_sec} seconds
Target word count: {int(shot.duration_sec * 2.5)} words (approx 2.5 words/sec)

Write ONLY the narration text, no stage directions.
```

### Step 2: TTS Synthesis (Edge-TTS)

```python
import edge_tts
import asyncio

async def synthesize(text: str, output_path: Path, voice: str = "en-US-GuyNeural") -> Path:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))
    return output_path
```

### Step 3: Duration Adjustment

If synthesized audio duration ≠ shot duration, apply speed adjustment via `ffmpeg`:

```python
def adjust_speed(audio_path: Path, target_duration: float) -> Path:
    actual_duration = get_audio_duration(audio_path)
    speed_factor = actual_duration / target_duration  # > 1 = faster
    speed_factor = max(0.75, min(1.25, speed_factor))  # clamp
    output_path = audio_path.with_suffix(".adj.wav")
    subprocess.run([
        "ffmpeg", "-i", str(audio_path),
        "-filter:a", f"atempo={speed_factor}",
        str(output_path)
    ], check=True)
    return output_path
```

### Output

```
outputs/{session_id}/audio/
    s1_sh1.wav
    s1_sh2.wav
    ...
```

---

## Stage 7: Post Agent

**File**: `src/agents/post.py`  
**Purpose**: Assemble all clips and audio into the final video.

### Input

```python
state.script_plan.scenes           # ordered scenes → ordered shots
shot.video_clip_path               # per shot
shot.audio_clip_path               # per shot (optional)
shot.transition_out                # "cut", "fade_to_black", "dissolve"
shot.duration_sec
```

### Assembly Logic (MoviePy)

```python
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip
)

def assemble(self, state: PipelineState) -> Path:
    clips = []
    shots = state.get_all_shots_ordered()
    
    for i, shot in enumerate(shots):
        clip = VideoFileClip(shot.video_clip_path)
        
        # Attach audio if available
        if shot.audio_clip_path:
            audio = AudioFileClip(shot.audio_clip_path)
            clip = clip.set_audio(audio)
        
        # Apply transition (outgoing)
        if shot.transition_out == "fade_to_black":
            clip = clip.fadeout(0.5)
        elif shot.transition_out == "dissolve" and i < len(shots) - 1:
            # Handle in concatenation
            pass
        
        clips.append(clip)
    
    # Concatenate with crossfade for dissolve transitions
    final = concatenate_videoclips(clips, method="compose")
    
    output_path = Path(state.output_dir) / "final_video.mp4"
    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
    )
    return output_path
```

### Output

```
outputs/{session_id}/final_video.mp4    # full video, 1080p, H.264
```

---

## Checkpoint System

**File**: `src/pipeline/checkpoint.py`

State is serialized to JSON after each stage:

```python
def save_checkpoint(state: PipelineState, stage: str) -> Path:
    path = Path("checkpoints") / state.session_id / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))
    return path

def load_checkpoint(session_id: str, stage: str) -> PipelineState:
    path = Path("checkpoints") / session_id / f"{stage}.json"
    return PipelineState.model_validate_json(path.read_text())
```

Resume from checkpoint:
```bash
python main.py --resume <session_id>
# Automatically detects last completed stage and continues
```

---

## Config Reference (`config.yaml`)

```yaml
deepseek:
  api_key: ${DEEPSEEK_API_KEY}
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 4096

flux:
  model_id: "black-forest-labs/FLUX.1-dev"
  num_inference_steps: 20
  guidance_scale: 3.5
  width: 1024
  height: 576
  dtype: "bfloat16"

rife:
  executable: "rife-ncnn-vulkan/rife-ncnn-vulkan.exe"
  model: "rife-v4.6"
  interpolation_factor: 4
  gpu_id: 0

tts:
  enabled: true
  voice: "en-US-GuyNeural"
  speed_clamp: [0.75, 1.25]

video:
  target_fps: 24
  output_codec: "libx264"
  output_bitrate: "8000k"
  resolution: [1920, 1080]

consistency:
  character_threshold: 0.80
  scene_threshold: 0.70
  max_retries_per_shot: 3
  use_deepseek_vision: true

pipeline:
  checkpoint_dir: "checkpoints"
  output_dir: "outputs"
  max_scenes: 8
  max_shots_per_scene: 3
```
