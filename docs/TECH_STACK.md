# Tech Stack: StoryForge

## Hardware Requirements

| Component | Minimum | Recommended (our setup) |
|-----------|---------|-------------------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| VRAM | 20GB | 24GB |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 100GB+ (models + outputs) |
| CUDA | 11.8 | 12.1+ |

---

## Software Stack

### Language & Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| CUDA Toolkit | 12.1 | GPU compute |
| cuDNN | 8.9+ | Deep learning acceleration |

---

### Core ML Libraries

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.4.0+cu121 | PyTorch core |
| `torchvision` | 0.19.0 | Vision utilities |

---

### Image Generation: FLUX.1-dev

```bash
pip install diffusers>=0.30.0 transformers>=4.44.0 accelerate>=0.33.0
pip install sentencepiece protobuf
```

| Package | Version | Purpose |
|---------|---------|---------|
| `diffusers` | ≥0.30.0 | FLUX.1-dev pipeline |
| `transformers` | ≥4.44.0 | Text encoder (T5, CLIP) |
| `accelerate` | ≥0.33.0 | Device placement |

**Model download** (~24GB, one-time):
```bash
huggingface-cli login   # needs HF account + model access request
python -c "
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')
"
# Cached to ~/.cache/huggingface/hub/
```

**Alternative (no login required)**:
```bash
# FLUX.1-schnell: open weights, 4-step, faster but lower quality
python -c "
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')
"
```

---

### Consistency Checking: CLIP

```bash
pip install open-clip-torch>=2.24.0
```

| Package | Version | Purpose |
|---------|---------|---------|
| `open-clip-torch` | ≥2.24.0 | CLIP-L/14 embeddings |

**Model download** (~900MB, auto on first use):
```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
```

---

### Frame Interpolation: RIFE

RIFE runs as a pre-compiled binary (rife-ncnn-vulkan). No Python install needed.

**Download** (Windows):
```
https://github.com/nihui/rife-ncnn-vulkan/releases
→ rife-ncnn-vulkan-20221029-windows.zip
→ Extract to: StoryForge/rife-ncnn-vulkan/
```

Models bundled in the zip. Use `rife-v4.6` (best quality/speed balance).

**Linux**:
```bash
wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip rife-ncnn-vulkan-20221029-ubuntu.zip
chmod +x rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan
```

**Verify**:
```bash
./rife-ncnn-vulkan/rife-ncnn-vulkan.exe -h
```

---

### LLM Inference: DeepSeek API

```bash
pip install openai>=1.40.0   # DeepSeek uses OpenAI-compatible API
```

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.40.0 | DeepSeek API client |

**API setup**:
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",           # from platform.deepseek.com
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "..."}],
    response_format={"type": "json_object"},  # structured output
    temperature=0.7,
)
```

**Models available**:
- `deepseek-chat` — DeepSeek-V3, general reasoning (use for all text agents)
- `deepseek-reasoner` — R1, slower but better for complex planning (optional for Director)

**Cost estimate** (~1 million tokens/month free tier as of 2025):
- Director + Cinematographer: ~2000 tokens per video
- Consistency (if vision enabled): ~500 tokens per shot × 15 shots = ~7500 tokens
- Narrator: ~1000 tokens per video
- **Total per video**: ~12,000 tokens — essentially free on free tier

---

### TTS: Edge-TTS

```bash
pip install edge-tts>=6.1.10
```

| Package | Version | Purpose |
|---------|---------|---------|
| `edge-tts` | ≥6.1.10 | Microsoft Edge TTS (free, no API key) |

**Usage**:
```python
import edge_tts, asyncio

async def synth(text: str, output: str, voice: str = "en-US-GuyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output)

asyncio.run(synth("The fox crossed the clearing.", "audio.mp3"))
```

**Requires internet connection** (Edge TTS is cloud-based but free).

**Local alternative** (fully offline):
```bash
pip install kokoro-onnx    # ~300MB model, high quality
```

---

### Video Processing

```bash
pip install moviepy==1.0.3
pip install opencv-python>=4.10.0
```

| Package | Version | Purpose |
|---------|---------|---------|
| `moviepy` | 1.0.3 | Video assembly, transitions, export |
| `opencv-python` | ≥4.10.0 | Frame I/O, VideoWriter |
| `ffmpeg` | system install | Required by MoviePy backend |

**ffmpeg install**:
```bash
# Windows (via winget):
winget install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH: C:\ffmpeg\bin\
```

---

### Data Validation & Utilities

```bash
pip install pydantic>=2.8.0
pip install pyyaml>=6.0.2
pip install pillow>=10.4.0
pip install numpy>=1.26.0
pip install scipy>=1.13.0
pip install tqdm>=4.66.0
pip install rich>=13.7.0
pip install typer>=0.12.0
```

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | ≥2.8.0 | Data model validation |
| `pyyaml` | ≥6.0.2 | Config loading |
| `pillow` | ≥10.4.0 | Image I/O |
| `numpy` | ≥1.26.0 | CLIP embedding math |
| `scipy` | ≥1.13.0 | Cosine similarity |
| `tqdm` | ≥4.66.0 | Progress bars |
| `rich` | ≥13.7.0 | Pretty console output |
| `typer` | ≥0.12.0 | CLI argument parsing |

---

## Full `requirements.txt`

```txt
# PyTorch (install separately with CUDA index URL above)
# torch==2.4.0
# torchvision==0.19.0

# FLUX image generation
diffusers>=0.30.0
transformers>=4.44.0
accelerate>=0.33.0
sentencepiece
protobuf

# CLIP consistency
open-clip-torch>=2.24.0

# LLM API
openai>=1.40.0

# TTS
edge-tts>=6.1.10

# Video processing
moviepy==1.0.3
opencv-python>=4.10.0

# Data & utilities
pydantic>=2.8.0
pyyaml>=6.0.2
pillow>=10.4.0
numpy>=1.26.0
scipy>=1.13.0
tqdm>=4.66.0
rich>=13.7.0
typer>=0.12.0
```

---

## Environment Setup (Full Walkthrough)

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# 2. Install PyTorch with CUDA
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other deps
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# → NVIDIA GeForce RTX 4090

# 5. Set up API key (Windows PowerShell)
$env:DEEPSEEK_API_KEY = "sk-..."

# 6. HuggingFace login (for FLUX.1-dev)
pip install huggingface-hub
huggingface-cli login

# 7. Pre-download models (run once)
python scripts/download_models.py

# 8. Verify RIFE binary
.\rife-ncnn-vulkan\rife-ncnn-vulkan.exe -h

# 9. Test the pipeline with a short story
python main.py --story "A cat sits by a window watching rain." --duration 10 --dry-run
```

---

## VRAM Budget at Runtime

```
Stage 3 (FLUX generation):
  FLUX.1-dev (bfloat16):    ~14 GB
  Safety checker:            disabled
  Available for OS:          ~8 GB
  Total used:                ~14-16 GB / 24 GB ✓

Stage 4 (CLIP check):
  CLIP-L/14:                 ~1 GB
  (FLUX offloaded to CPU)
  Total used:                ~2-3 GB / 24 GB ✓

Stage 5 (RIFE interpolation):
  RIFE model:                ~2-4 GB
  (CLIP unloaded)
  Total used:                ~4 GB / 24 GB ✓
```

No OOM risk on 4090. Models are never loaded simultaneously.

---

## Model Storage Summary

| Model | Size | Location |
|-------|------|----------|
| FLUX.1-dev | ~24 GB | `~/.cache/huggingface/hub/` |
| CLIP ViT-L/14 | ~900 MB | `~/.cache/huggingface/hub/` |
| RIFE v4.6 | ~20 MB | `./rife-ncnn-vulkan/models/` |
| Edge-TTS | 0 (streaming) | cloud |
| DeepSeek | 0 (API) | cloud |

**Total local storage needed**: ~26 GB for models + space for outputs.
