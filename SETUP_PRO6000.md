# StoryForge — Setup Runbook for AutoDL RTX PRO 6000

> Use this when starting a fresh PRO 6000 instance. Assumes you've already
> uploaded the code via rsync; if not, see Step 0.

**Instance details (this rental)**:
- SSH: `ssh -p 41139 root@connect.cqa1.seetacloud.com`
- Region: cqa1 (Chongqing A1)
- Data disk: `/root/autodl-tmp/` (80 GB)

The PRO 6000 has 96 GB VRAM and Blackwell architecture (sm_120). This means:
- **No CPU offload needed** for FLUX-dev — load it fully on GPU for max speed.
- **PyTorch must be 2.5+ with CUDA 12.4+** — older PyTorch (2.4 / cu121) does
  NOT support Blackwell and will fail to detect the GPU.

---

## Step 0 — Upload code from local (Windows Git Bash)

```bash
rsync -avzP -e "ssh -p 41139" \
    --exclude='outputs/' --exclude='checkpoints/' --exclude='__pycache__/' \
    --exclude='.venv/' --exclude='Project3说明.pdf' --exclude='*.pyc' \
    /d/CS/Graphics/PJ3/ \
    root@connect.cqa1.seetacloud.com:/root/autodl-tmp/storyforge/
```

If rsync isn't available, use scp:
```bash
scp -P 41139 -r /d/CS/Graphics/PJ3 \
    root@connect.cqa1.seetacloud.com:/root/autodl-tmp/storyforge_upload
# Then on remote: mv /root/autodl-tmp/storyforge_upload /root/autodl-tmp/storyforge
```

---

## Step 1 — SSH in and enable academic-network proxy

```bash
ssh -p 41139 root@connect.cqa1.seetacloud.com
cd /root/autodl-tmp/storyforge
source /etc/network_turbo            # AutoDL proxy for HF / GitHub access
```

Verify GPU is visible to nvidia-smi:
```bash
nvidia-smi
# Expect: NVIDIA RTX PRO 6000 Blackwell, 96GB
# Driver version should be 570+ for Blackwell support
```

---

## Step 2 — Create a fresh venv (DO NOT reuse the 4090 venv)

The old venv has `torch==2.4.0+cu121` which won't run on Blackwell.

```bash
python3 -m venv /root/autodl-tmp/venvs/storyforge_pro6k
source /root/autodl-tmp/venvs/storyforge_pro6k/bin/activate
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.cache-dir /root/autodl-tmp/.pipcache
```

---

## Step 3 — Install PyTorch with CUDA 12.4 for Blackwell

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu124

# Verify Blackwell GPU is detected:
python -c "
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
"
# Expect:
#   Torch: 2.5.1+cu124
#   CUDA available: True
#   GPU: NVIDIA RTX PRO 6000 Blackwell
#   Compute capability: (12, 0)  ← sm_120
```

If GPU is **NOT detected** with cu124, try the nightly build with cu126:
```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

---

## Step 4 — Install the rest

```bash
cd /root/autodl-tmp/storyforge
pip install -r requirements.txt

# Critical follow-up: remove flash-attn / kernels which break with diffusers 0.31
pip uninstall -y flash-attn flash_attn kernels 2>/dev/null

# System tools
apt update && apt install -y ffmpeg unzip libvulkan1

# Verify the diffusers FLUX pipeline imports cleanly:
python -c "from diffusers import FluxPipeline; print('FluxPipeline OK')"
```

---

## Step 5 — Download RIFE binary

```bash
cd /root/autodl-tmp/storyforge

wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip rife-ncnn-vulkan-20221029-ubuntu.zip
mv rife-ncnn-vulkan-20221029-ubuntu rife-ncnn-vulkan
chmod +x rife-ncnn-vulkan/rife-ncnn-vulkan
./rife-ncnn-vulkan/rife-ncnn-vulkan -h | head -5
```

---

## Step 6 — Set up HF cache and download FLUX

**Use the standard cache layout. Do NOT override HF_HUB_CACHE manually** — let
it default to `$HF_HOME/hub`. (This was the source of yesterday's pain.)

```bash
# Add to ~/.bashrc (once):
cat >> ~/.bashrc <<'EOF'

# StoryForge runtime env
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIP_DISABLE_PIP_VERSION_CHECK=1
EOF

source ~/.bashrc

# Confirm only HF_HOME is set, HF_HUB_CACHE is empty (will default to $HF_HOME/hub):
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"   # should be empty

# Log into HF (accept FLUX.1-dev license in browser first):
huggingface-cli login

# Pre-download FLUX (~24 GB). Takes 10-20 min depending on bandwidth:
hf download black-forest-labs/FLUX.1-dev
```

Verify the cache:
```bash
python -c "
from huggingface_hub import scan_cache_dir
for r in scan_cache_dir().repos:
    if 'FLUX' in r.repo_id:
        print(r.repo_id, '→', r.size_on_disk_str)
"
# Expect: black-forest-labs/FLUX.1-dev → ~32 GB
```

---

## Step 7 — Set secrets

```bash
cd /root/autodl-tmp/storyforge
cp .env.example .env
nano .env
# Fill in:
#   DEEPSEEK_API_KEY=sk-...
#   HF_TOKEN=hf_...
# (DO NOT add HF_HUB_CACHE here — it should not be set)
```

---

## Step 8 — Sanity check

```bash
cd /root/autodl-tmp/storyforge
source /root/autodl-tmp/venvs/storyforge_pro6k/bin/activate
set -a && source .env && set +a

# Director-only dry-run (no GPU, ~10s):
python main.py dry-run "A fox crosses a snowy forest at dusk." --duration 15
# Expect: valid JSON ScriptPlan printed in green

# Confirm GPU sees FLUX without re-downloading:
python -c "
from diffusers import FluxPipeline
import torch
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)
pipe.to('cuda')
print('FLUX on GPU, no download. VRAM:', torch.cuda.memory_allocated() / 1e9, 'GB')
"
# Expect: ~24 GB allocated, no download bars
```

---

## Step 9 — Full run

```bash
tmux new -s run

# Inside tmux:
cd /root/autodl-tmp/storyforge
source /root/autodl-tmp/venvs/storyforge_pro6k/bin/activate
set -a && source .env && set +a

# 5-second test first (proves the pipeline, ~3-4 min):
python main.py run "A red fox stands in falling snow." --duration 5 \
    --style "cinematic, photorealistic"

# If green, do the real 15s run:
python main.py run \
    "A fox crosses a snowy forest at dusk, pauses to look at the moon, then disappears." \
    --duration 20 \
    --style "cinematic, photorealistic, golden hour"
```

Detach from tmux: `Ctrl+B` then `D`. Reattach: `tmux attach -t run`.

---

## Expected timing on PRO 6000

| Stage | Time |
|---|---:|
| 1+2 (LLM) | ~30 s |
| 3 (FLUX, no offload, ~95 keyframes for 20s × ~4s each) | **~6 min** |
| 4 (Consistency) | ~1 min |
| 5 (RIFE + encode) | ~1 min |
| 6 (Narrator) | ~1 min |
| 7 (Post) | ~30 s |
| **Total for 20s video** | **~10 min** |

---

## Pulling results back to local

```bash
# In local Git Bash:
ssh -p 41139 root@connect.cqa1.seetacloud.com \
    "ls /root/autodl-tmp/storyforge/outputs/"
# Pick a session id, then:
mkdir -p /d/CS/Graphics/PJ3/results
scp -P 41139 \
    root@connect.cqa1.seetacloud.com:/root/autodl-tmp/storyforge/outputs/<SID>/final_video.mp4 \
    /d/CS/Graphics/PJ3/results/
```

---

## Common issues + fixes

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` returns False | PyTorch built for older arch | Reinstall with cu124 or cu126 nightly |
| `unsupported compute capability sm_120` | Same | Same |
| FLUX downloads again on launch | HF_HUB_CACHE set incorrectly | `unset HF_HUB_CACHE` — let it default |
| RIFE: no Vulkan device | Missing libvulkan1 | `apt install libvulkan1 vulkan-tools` |
| OOM at 96 GB (??) | Memory leak from old run | `pkill -9 python` + `nvidia-smi` to confirm 0 MiB |

---

## What changed vs. the 4090 setup

1. **No CPU offload needed**: `offload_mode: none` in config.yaml — 5-10x faster
2. **PyTorch 2.5.1 + cu124** (not 2.4.0 + cu121) — Blackwell requires it
3. **No more sequential offload pain** — model fits, all features stay on GPU
4. **Higher resolutions are an option** — try `width: 1280, height: 720` for HD

The code itself (agents, orchestrator, prompts) is unchanged — only the deployment
environment and config differ.
