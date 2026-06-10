# StoryForge: Multi-Agent Text-to-Video Generation Pipeline

> 2026 Spring Computer Graphics — Project 3
> Direction: Video Generation
> Topic: Multi-Agent Collaborative Text-to-Video Synthesis with Cross-Shot Consistency

## Overview

StoryForge is a locally-deployable multi-agent pipeline that converts natural
language story descriptions into short animated videos. Seven specialized
agents collaborate through a structured shared state, each responsible for
one stage of production — from script breakdown to final video export.

**Key claim**: A consistency-gated cascade in which **CLIP-approved FLUX
keyframes seed CogVideoX image-to-video diffusion**, jointly solving the two
fundamental failure modes of multi-shot AI video — *within-shot motion
incoherence* and *cross-shot character drift* — with no proprietary closed-
source generation model required. DeepSeek handles all language reasoning;
FLUX.1-dev handles keyframe synthesis; CogVideoX-5b-I2V handles motion
synthesis; all visual inference runs locally on a single workstation GPU.

```
"A fox crosses a snowy forest at dusk, pauses, looks at the moon, then disappears."
                                        │
                                        ▼
                            [ StoryForge Pipeline ]
                                        │
                                        ▼
                            fox_in_snow_final.mp4  (~18 s, 1080p)
```

---

## Repository Structure

```
StoryForge/
├── README.md                       # this file
├── PROJECT.md                      # master design reference (start here)
├── SETUP_PRO6000.md                # remote deployment runbook
├── requirements.txt
├── config.yaml
├── .env.example
├── main.py                         # Typer CLI
│
├── docs/                           # supplementary docs (subsumed by PROJECT.md)
│   ├── ARCHITECTURE.md
│   ├── PIPELINE.md
│   ├── AGENTS.md
│   ├── TECH_STACK.md
│   └── REPORT_OUTLINE.md           # academic paper structure
│
├── src/
│   ├── schemas/data_models.py      # Pydantic models (the shared state)
│   ├── utils/                      # LLM client, prompts, ffmpeg helpers
│   ├── models/
│   │   ├── flux_wrapper.py         # FLUX/SDXL pipeline wrapper
│   │   ├── clip_wrapper.py         # CLIP ViT-L/14 for consistency
│   │   ├── cogvideox_wrapper.py    # CogVideoX-5b-I2V — motion synthesizer
│   │   └── rife_wrapper.py         # legacy RIFE (opt-in via FORCE_RIFE=1)
│   ├── pipeline/                   # orchestrator + checkpointing
│   └── agents/                     # 7 specialized agents
│
├── rife-ncnn-vulkan/               # optional; only if FORCE_RIFE=1
├── outputs/<session_id>/           # gitignored
└── checkpoints/<session_id>/       # gitignored
```

---

## Pipeline at a Glance

| # | Agent | Model | Output |
|---|---|---|---|
| 1 | **Director** | DeepSeek-V3 | Structured `ScriptPlan` (characters, scenes, shots) |
| 2 | **Cinematographer** | DeepSeek-V3 | Per-shot FLUX prompt + negative prompt |
| 3 | **Visual** | FLUX.1-dev | 1 reference keyframe per shot (1024×576) |
| 4 | **Consistency** | CLIP ViT-L/14 (+ optional DeepSeek vision) | Pass/fail gate on character identity |
| 5 | **Animator** | **CogVideoX-5b-I2V** | 6.1 s motion video per shot (720×480) |
| 6 | **Narrator** *(optional)* | DeepSeek + Edge-TTS | Per-shot narration audio |
| 7 | **Post** | MoviePy + ffmpeg | Final 1080p H.264 video with cross-dissolves |

**Stage 3 ⇄ Stage 4** is a retry loop: if the consistency gate fails, the
Cinematographer rewrites the prompt and the keyframe regenerates (max 3 retries).

The shared state is a Pydantic v2 `PipelineState` object. Per-stage
serialization to JSON gives free checkpoint-based resumability.

---

## Quick Start

### Hardware
- NVIDIA GPU with ≥48 GB VRAM (96 GB recommended for full speed, no offload)
- ~70 GB free disk for model weights + venv + outputs

### Software setup
```bash
git clone <repo> storyforge && cd storyforge
python3 -m venv .venv && source .venv/bin/activate

# PyTorch (use cu126 nightly for Blackwell GPUs like RTX 5090/PRO 6000; cu124 for Ada/Ampere)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

pip install -r requirements.txt
pip uninstall -y flash-attn flash_attn kernels   # if present, breaks diffusers 0.31
pip install hf_transfer                          # fast HF downloads

apt install -y ffmpeg
```

### Secrets
```bash
cp .env.example .env
# Edit .env to fill in:
#   DEEPSEEK_API_KEY=sk-...
#   HF_TOKEN=hf_...
```

### Model weights
```bash
# Accept licenses first (in browser, signed in to your HF account):
#   https://huggingface.co/black-forest-labs/FLUX.1-dev
#   https://huggingface.co/THUDM/CogVideoX-5b-I2V

export HF_HOME=/path/to/data_disk/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli login
hf download black-forest-labs/FLUX.1-dev
hf download THUDM/CogVideoX-5b-I2V
```

### First run
```bash
set -a && source .env && set +a

python main.py run \
    "A red fox stands in a snowy forest at dusk." \
    --duration 12 \
    --style "cinematic, photorealistic"
```

Output: `outputs/<session_id>/final_video.mp4`

### Useful commands
```bash
python main.py dry-run "..."              # Director-only, no GPU
python main.py list-sessions              # list checkpointed runs
python main.py resume <session_id>        # resume after interrupt
```

---

## Core Technical Contribution

Pure text-to-image + interpolation pipelines (e.g. FLUX + RIFE) can preserve
identity within a shot but produce distorted in-betweens when keyframes
don't share motion. Pure text-to-video models generate motion but cannot
enforce specific visual identity across shots from the prompt alone.

**StoryForge separates the two concerns:**

- **Identity gate (FLUX + CLIP)**: each shot's first frame is generated by
  FLUX.1-dev and gated against a frozen per-character reference embedding.
  If similarity falls below threshold (default 0.80), the prompt is refined
  and the frame is regenerated.

- **Motion synthesis (CogVideoX-I2V)**: the approved first frame conditions
  CogVideoX, which generates 6.1 seconds of coherent motion from that frame
  + the shot's text prompt. Identity propagates into the motion automatically
  because CogVideoX is trained to preserve appearance from the conditioning image.

See **PROJECT.md §6** for the deep dive and §18 for the historical evolution
from the original RIFE-based design.

---

## Performance (96 GB PRO 6000, no offload)

| Output | Time |
|---|---:|
| 6 s test (1 shot) | ~5 min |
| 12 s video (2 shots) | ~9 min |
| 18 s video (3 shots) | ~13 min |
| 36 s video (6 shots) | ~22 min |

Stage 5 (CogVideoX) dominates at ~3 min per shot. See PROJECT.md §11 for the
full breakdown and §10 for VRAM budgets on smaller GPUs.

---

## Configuration

Everything is in `config.yaml`. Defaults are tuned for a 96 GB PRO 6000
Blackwell. See PROJECT.md §9 for an annotated reference and a tuning playbook.

Key sections:
```yaml
flux:                      # keyframe synthesizer
  model_id: black-forest-labs/FLUX.1-dev
  offload_mode: none       # "none" | "model" | "sequential"

cogvideox:                 # motion synthesizer (Stage 5)
  model_id: THUDM/CogVideoX-5b-I2V
  num_inference_steps: 50
  num_frames: 49
  offload_mode: none

consistency:
  character_threshold: 0.80
  max_retries_per_shot: 3

pipeline:
  max_scenes: 3
  max_shots_per_scene: 2
```

---

## Citation / Acknowledgments

This project uses:
- **DeepSeek-V3** (DeepSeek-AI) for language reasoning
- **FLUX.1-dev** (Black Forest Labs) for keyframe synthesis
- **CogVideoX-5b-I2V** (Tsinghua University) for motion synthesis
- **CLIP ViT-L/14** (OpenAI / open_clip) for consistency embedding
- **Edge-TTS** (Microsoft) for optional narration

All design, implementation, and writing by the project author. Engineering
assistance from Claude Code (Anthropic).

See PROJECT.md §17 for known limitations and future work.

---

## License

Code: MIT. Model weights: as per each model's own license.
