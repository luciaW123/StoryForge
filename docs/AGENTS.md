# Agent Specifications: StoryForge

Each agent is a Python class with the interface:
```python
class BaseAgent:
    def process(self, state: PipelineState) -> PipelineState: ...
```

---

## Agent 1: Director

**File**: `src/agents/director.py`  
**Role**: Story analyst and screenplay writer.

### Responsibility
Convert unstructured story text into a structured `ScriptPlan`. This is the most creative agent — it makes narrative decisions (pacing, scene division, character extraction) that all downstream agents inherit.

### Inputs
- `state.story_input` (str) — raw user story
- `config.pipeline.max_scenes` (int) — upper bound on scenes
- CLI args: `target_duration`, `style_hint`

### Processing Logic
```
1. LLM call: story → ScriptPlan JSON (structured output mode)
2. Validate JSON against ScriptPlan Pydantic schema
3. If invalid → retry (up to 3x) with correction prompt
4. Rescale shot durations if sum ≠ target_duration
5. Assign reference_seed to each character (hash of character name)
6. Save checkpoint: "scripting"
```

### Prompt Strategy
Uses **structured output** (JSON mode) with DeepSeek. The system prompt includes the full ScriptPlan JSON schema as a reference. Temperature: 0.8 (slightly creative for narrative decisions).

### Failure Modes & Recovery
| Failure | Recovery |
|---------|----------|
| JSON parse error | Retry with "respond ONLY with valid JSON" constraint |
| > max_scenes scenes | Ask LLM to consolidate smallest scenes |
| Duration mismatch > 20% | Rescale proportionally without LLM |
| No characters extracted | Inject a default "narrator perspective" character |

### Output State Changes
```python
state.script_plan = ScriptPlan(...)
state.stage = "PLANNING"
```

---

## Agent 2: Cinematographer

**File**: `src/agents/cinematographer.py`  
**Role**: Visual architect and FLUX prompt engineer.

### Responsibility
Transform abstract shot descriptions into FLUX-ready prompts. Also responsible for **prompt refinement** when the Consistency Agent requests a retry.

### Inputs
- `state.script_plan` — complete ScriptPlan with shot descriptions
- Each `Shot.description`, `Shot.camera_angle`, `Shot.camera_movement`
- Each `Character.visual_prompt` for characters in the shot

### Processing Logic
```
1. For each shot (can batch 5 shots per API call):
   a. Construct user prompt with shot + character context
   b. LLM call → {flux_prompt, negative_prompt, num_keyframes}
   c. Append global_style_suffix to flux_prompt
   d. Validate num_keyframes matches duration formula
2. Save checkpoint: "planning"
```

### Prompt Construction Rules
The Cinematographer follows these rules when writing FLUX prompts:
1. **Subject first**: Always start with the main subject/character
2. **Action**: What is happening in this frame
3. **Environment**: Setting details, spatial context
4. **Lighting**: Light source, quality, time-of-day signals
5. **Atmosphere**: Weather, particles (snow, dust, fog)
6. **Camera**: Lens, angle, depth of field
7. **No style tokens**: These come from `global_style_suffix`

**Good example**:
```
"a red fox with amber fur standing at the edge of a snow-covered pine forest clearing,
 one paw raised mid-step, soft golden dusk light through pine trees casting long shadows,
 gentle snowfall catching light, breath mist visible, shallow depth of field,
 foreground snow bokeh, eye-level camera angle"
```

**Bad example** (too abstract, no visual specifics):
```
"fox in forest, beautiful, artistic" 
```

### Refinement Sub-method

Called when Consistency Agent returns FAIL:
```python
def refine_prompt(self, shot: Shot, feedback: str, state: PipelineState) -> str:
```

Refinement prompt:
```
You are refining a FLUX image generation prompt that failed a consistency check.

Original prompt: {shot.flux_prompt}
Failure feedback: {feedback}
Character visual reference: {character.visual_prompt}
Shot description intent: {shot.description}

Rewrite the prompt to:
1. Fix the specific consistency issue
2. Make the character description more prominent
3. Keep the scene intent intact

Respond with ONLY the new flux_prompt string (no JSON wrapper).
```

### Output State Changes
```python
for shot in state.all_shots():
    shot.flux_prompt = "..."
    shot.negative_prompt = "..."
    shot.num_keyframes = N
state.stage = "GENERATING"
```

---

## Agent 3: Visual

**File**: `src/agents/visual.py`  
**Role**: Image generator. The only agent that uses GPU compute intensively.

### Responsibility
Generate keyframe images for each shot using FLUX.1-dev. Manages VRAM budget and batching.

### Inputs (per shot)
- `shot.flux_prompt` (str)
- `shot.negative_prompt` (str)
- `shot.num_keyframes` (int)
- `character.reference_seed` (int) — for character consistency via seeding

### Processing Logic
```
1. Load FluxWrapper (singleton, loaded once at agent init)
2. For each shot:
   a. Determine seed strategy:
      - Solo character shot → fixed seed (reference_seed)
      - Multi-char or no-char → random seed per keyframe (±1 increments)
   b. Generate num_keyframes images, one at a time
   c. Save to outputs/{session_id}/keyframes/{shot.id}_kf{N:03d}.png
   d. Update shot.keyframe_paths
3. No checkpoint here (images on disk serve as checkpoint)
```

### VRAM Management
```python
# Before generation: free CLIP from VRAM if loaded
# After generation: keep FLUX loaded (next shot needs it immediately)
# Between Visual and Consistency stages: offload FLUX, load CLIP

# In FluxWrapper:
pipe.enable_model_cpu_offload()   # keeps model on CPU, moves to GPU per-layer
# OR
pipe.enable_sequential_cpu_offload()  # even more VRAM-efficient, slower
```

### Seed Strategy Details
```python
def get_seeds_for_shot(self, shot: Shot, characters: list[Character]) -> list[int]:
    if len(characters) == 1:
        base = characters[0].reference_seed
    else:
        base = hash(shot.id) % (2**31)
    
    if shot.camera_movement == "static":
        # Same seed for all frames → minimal variation
        return [base] * shot.num_keyframes
    else:
        # Slight drift → natural motion appearance
        return [base + i for i in range(shot.num_keyframes)]
```

### Output State Changes
```python
shot.keyframe_paths = [Path(...), ...]
state.stage = "CHECKING"
```

---

## Agent 4: Consistency

**File**: `src/agents/consistency.py`  
**Role**: Quality gatekeeper. The only agent that can reject output and trigger retry.

### Responsibility
Score visual quality of keyframes on two dimensions: character identity consistency, and text-image alignment. Return PASS/FAIL with structured feedback for refinement.

### Inputs (per shot)
- `shot.keyframe_paths`
- `shot.characters_present`
- `character.clip_embedding` (reference, may be None for first appearance)
- `shot.description` (for text-alignment check)

### Processing Logic
```
1. For each shot:
   a. Load keyframe images
   b. Extract CLIP embeddings
   c. Character consistency check (Level 1):
      - If char has no reference: embed first keyframe, store as reference → PASS
      - If char has reference: compute cosine similarity → PASS if ≥ 0.80
   d. Text-image alignment check (Level 2):
      - DeepSeek vision API call (if enabled in config)
      - Send first keyframe as base64 + shot description
      - LLM returns YES/NO + one-line feedback
   e. If both pass: shot.approved = True
   f. If fail: construct feedback string, return to orchestrator
2. Scene continuity advisory check (Level 3, no-block):
   - Log warnings if consecutive shots in same scene diverge > threshold
```

### CLIP Model Setup
```python
# src/models/clip_wrapper.py
import open_clip

class CLIPWrapper:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k'
        )
        self.model.eval().cuda()
    
    def encode_image(self, image: PIL.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return emb.cpu().numpy()[0]
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### Feedback String Format

When returning FAIL, feedback is structured for the Cinematographer:
```
"CHARACTER_INCONSISTENCY: The fox's fur color appears inconsistent 
 (similarity=0.71, threshold=0.80). Strengthen visual description:
 add 'vibrant amber fur, white-tipped tail' more prominently.
 Original character reference: {character.visual_prompt}"
```

### Retry Orchestration

The Orchestrator handles the retry loop; Consistency Agent only scores:
```python
# In orchestrator.py:
for shot in state.all_shots():
    while not shot.approved and shot.retry_count < MAX_RETRIES:
        state = self.visual.generate_shot(state, shot)
        passed, feedback = self.consistency.evaluate_shot(state, shot)
        if not passed:
            shot.flux_prompt = self.cinematographer.refine_prompt(shot, feedback, state)
            shot.retry_count += 1
        else:
            shot.approved = True
    
    if not shot.approved:
        logger.warning(f"Shot {shot.id} not approved after {MAX_RETRIES} retries. Using best attempt.")
        shot.approved = True  # force-approve, log quality score
```

### Output State Changes
```python
shot.consistency_score = 0.87
shot.approved = True
character.clip_embedding = [...]  # stored after first approval
state.stage = "INTERPOLATING"
```

---

## Agent 5: Animator

**File**: `src/agents/animator.py`  
**Role**: Temporal bridge — turns static keyframes into fluid motion.

### Responsibility
Run RIFE frame interpolation on each shot's keyframes, then assemble frames into per-shot video clips at target FPS.

### Inputs (per shot)
- `shot.keyframe_paths` — ordered list of PNG keyframes
- `config.rife.interpolation_factor` — 2 or 4
- `config.video.target_fps` — 24

### Processing Logic
```
1. For each shot:
   a. Copy keyframes to temp directory (rife-ncnn-vulkan expects flat dir)
   b. Run RIFE binary: keyframes_dir → interpolated_frames_dir
   c. Verify output frame count matches expected count
   d. Assemble frames into MP4 using OpenCV VideoWriter
   e. Verify output duration matches shot.duration_sec (±0.1s tolerance)
   f. Save to outputs/{session_id}/clips/{shot.id}.mp4
2. Save checkpoint: "interpolating"
```

### RIFE Binary Call
```python
# rife-ncnn-vulkan command syntax:
# ./rife-ncnn-vulkan -i input_dir -o output_dir -m model -n factor

def interpolate_shot(self, shot: Shot) -> Path:
    kf_dir = self._copy_keyframes_to_temp(shot)
    out_dir = self.temp_dir / f"{shot.id}_interp"
    
    result = subprocess.run([
        str(self.rife_exe),
        "-i", str(kf_dir),
        "-o", str(out_dir),
        "-m", self.model_name,
        "-n", str(self.factor),
        "-g", "0",
        "-j", "4:4:4",     # thread count: decode:interp:encode
    ], capture_output=True, text=True, check=True)
    
    return self._frames_to_mp4(out_dir, shot)
```

### Frame Count Validation
```python
expected_frames = shot.num_keyframes * self.factor - (self.factor - 1)
# RIFE n=4 on K keyframes → K + (K-1)*3 = 4K-3 frames
actual_frames = len(list(out_dir.glob("*.png")))
assert abs(actual_frames - expected_frames) <= 2, f"Frame count mismatch: {actual_frames} vs {expected_frames}"
```

### Output State Changes
```python
shot.video_clip_path = "outputs/.../clips/s1_sh1.mp4"
state.stage = "NARRATING"
```

---

## Agent 6: Narrator

**File**: `src/agents/narrator.py`  
**Role**: Scriptwriter and voice actor.

### Responsibility
Generate narration text for each shot, synthesize audio via TTS, and adjust playback speed to exactly match clip duration. Optional — disabled if `config.tts.enabled = false`.

### Inputs (per shot)
- `shot.description`
- `shot.duration_sec`
- `script_plan.overall_mood`
- `script_plan.title`

### Processing Logic
```
1. DeepSeek call: generate narration script for each shot
2. Edge-TTS: synthesize to WAV
3. Measure actual audio duration
4. If duration mismatch > 5%: apply atempo ffmpeg filter
5. Save to outputs/{session_id}/audio/{shot.id}.wav
6. Set shot.audio_clip_path
```

### Narration Script Generation

**Batch prompt** (all shots in one call for coherent voice):
```
You are writing voiceover narration for a short video titled "{title}".
Mood: {overall_mood}
Voice style: calm, descriptive, literary

Write a brief narration line for each shot:
{for each shot: "Shot {id} ({duration}s): {description}"}

Rules:
- Each line should be approximately {duration * 2.5} words
- Maintain narrative continuity across shots
- Do not describe camera movements or technical terms
- Write in present tense, third person

Respond with JSON: [{"shot_id": "s1_sh1", "narration": "..."}, ...]
```

### TTS Voices Available
```
en-US-GuyNeural       — neutral male, documentary
en-US-JennyNeural     — warm female, storytelling
en-GB-RyanNeural      — British male, formal
en-AU-NatashaNeural   — Australian female, nature docs
```

### Output State Changes
```python
shot.audio_clip_path = "outputs/.../audio/s1_sh1.wav"
state.stage = "EDITING"
```

---

## Agent 7: Post

**File**: `src/agents/post.py`  
**Role**: Film editor. Final assembly.

### Responsibility
Concatenate all shot clips in order, apply transitions, mix audio, optionally burn subtitles, and export the final video at full resolution.

### Inputs
- All `shot.video_clip_path` in order
- All `shot.audio_clip_path` (optional)
- All `shot.transition_out` types
- `config.video.resolution`, `config.video.target_fps`

### Processing Logic
```
1. Load all video clips via MoviePy
2. For each clip:
   a. Attach audio if available
   b. Resize to target resolution if needed
3. Apply transitions:
   - "cut": direct concatenation
   - "fade_to_black": clip.fadeout(0.5) + next_clip.fadein(0.5)
   - "dissolve": crossfade using CompositeVideoClip
4. Add optional title card (first 2s, text overlay)
5. Concatenate all clips
6. Write final MP4 with libx264 + aac
7. Save checkpoint: "done"
8. Print summary stats
```

### Transition Implementation
```python
def apply_transition(self, clip_a, clip_b, transition_type: str, duration: float = 0.5):
    if transition_type == "cut":
        return [clip_a, clip_b]
    
    elif transition_type == "fade_to_black":
        return [clip_a.fadeout(duration), clip_b.fadein(duration)]
    
    elif transition_type == "dissolve":
        # Crossfade: overlap last 0.5s of A with first 0.5s of B
        clip_a_trimmed = clip_a.subclip(0, clip_a.duration - duration)
        clip_b_trimmed = clip_b.subclip(duration)
        cross = CompositeVideoClip([
            clip_a.subclip(clip_a.duration - duration).set_start(clip_a_trimmed.duration),
            clip_b.subclip(0, duration)
                 .set_start(clip_a_trimmed.duration)
                 .crossfadein(duration),
        ])
        return [clip_a_trimmed, cross, clip_b_trimmed]
```

### Final Export Settings
```python
final.write_videofile(
    str(output_path),
    fps=24,
    codec="libx264",
    audio_codec="aac",
    bitrate="8000k",        # good quality at 1080p
    audio_bitrate="192k",
    preset="medium",        # encoding speed/size tradeoff
    ffmpeg_params=["-crf", "18"],
)
```

### Output State Changes
```python
state.final_video_path = "outputs/{session_id}/final_video.mp4"
state.stage = "DONE"
```

---

## Orchestrator

**File**: `src/pipeline/orchestrator.py`  
**Role**: Controller. Not an agent itself — coordinates agents and manages state machine transitions.

### Responsibilities
- Initialize all agents at startup
- Pass state between agents in correct order
- Manage the Stage 3→4 retry loop
- Save/load checkpoints
- Log timing and quality metrics
- Handle KeyboardInterrupt gracefully (save checkpoint before exit)

### Core Run Loop

```python
class PipelineOrchestrator:
    def run(self, story: str, duration: int, style: str) -> Path:
        state = self._init_state(story, duration, style)
        
        # Stage 1
        state = self.director.process(state)
        self._checkpoint(state, "scripting")
        
        # Stage 2
        state = self.cinematographer.process(state)
        self._checkpoint(state, "planning")
        
        # Stage 3+4: Generate + Check (with retry loop)
        for shot in state.all_shots():
            self._generate_and_check(shot, state)
        self._checkpoint(state, "generating")
        
        # Stage 5
        state = self.animator.process(state)
        self._checkpoint(state, "interpolating")
        
        # Stage 6
        if self.config.tts.enabled:
            state = self.narrator.process(state)
        self._checkpoint(state, "narrating")
        
        # Stage 7
        state = self.post.process(state)
        self._checkpoint(state, "done")
        
        self._print_summary(state)
        return Path(state.final_video_path)
    
    def _generate_and_check(self, shot: Shot, state: PipelineState) -> None:
        MAX_RETRIES = self.config.consistency.max_retries_per_shot
        while not shot.approved and shot.retry_count <= MAX_RETRIES:
            self.visual.generate_shot(state, shot)
            passed, feedback = self.consistency.evaluate_shot(state, shot)
            if not passed and shot.retry_count < MAX_RETRIES:
                shot.flux_prompt = self.cinematographer.refine_prompt(shot, feedback, state)
                shot.retry_count += 1
            else:
                shot.approved = True
        
        if not shot.approved:
            logger.warning(f"[{shot.id}] Max retries reached. Score: {shot.consistency_score:.2f}")
```
