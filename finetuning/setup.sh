#!/usr/bin/env bash
# setup.sh — Environment setup for drafter fine-tuning + acceptance measurement.
#
# Run this once on a fresh RunPod pod before using finetune_draft.py or
# measure_acceptance.py.  Mirrors dataset-generation/setup.sh and adds
# training-side dependencies.
#
# Usage:  bash finetuning/setup.sh
set -e

# ── Cache directories on RunPod volume (survives pod restarts) ────────────────
mkdir -p /workspace/hf_cache/hub
mkdir -p /workspace/hf_cache/transformers
mkdir -p /workspace/torch_cache
mkdir -p /workspace/pip_cache

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export TORCH_HOME=/workspace/torch_cache
export XDG_CACHE_HOME=/workspace/.cache
export PIP_CACHE_DIR=/workspace/pip_cache

# Faster HF downloads
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Clear any stale HF caches that can cause checksum errors
rm -rf /root/.cache/huggingface
rm -rf ~/.cache/huggingface
rm -rf /workspace/.cache/huggingface

# ── vLLM (required for measure_acceptance.py) ─────────────────────────────────
pip install "vllm==0.19.0" --break-system-packages

# ── Torch with CUDA 12.8 (overwrites the torch that vLLM pulled in) ──────────
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  --break-system-packages

# ── Training-side dependencies ─────────────────────────────────────────────────
pip install \
  "transformers>=4.45" \
  "accelerate>=1.0" \
  "peft>=0.12" \
  "bitsandbytes" \
  "datasets>=3.6.0" \
  "huggingface_hub>=0.30.0" \
  "hf_transfer>=0.1.8" \
  "sentencepiece>=0.2.0" \
  --break-system-packages

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python3 -c "import torch; print('torch :', torch.__version__, '| CUDA:', torch.version.cuda)"
python3 -c "from vllm import LLM, SamplingParams; print('vLLM  : ok')"
python3 -c "import transformers, peft; print('transformers:', transformers.__version__, '| peft:', peft.__version__)"
python3 -c "import accelerate; print('accelerate  :', accelerate.__version__)"
echo ""
echo "Setup complete. GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
