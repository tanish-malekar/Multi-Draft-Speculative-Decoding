#runpod template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Step 0: Put Hugging Face / Torch caches on the RunPod volume
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

# Optional but useful: avoid Xet reconstruction issues / temp disk pressure
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional cleanup from previous failed downloads
rm -rf /root/.cache/huggingface
rm -rf ~/.cache/huggingface
rm -rf /workspace/.cache/huggingface

# Step 1: Install vLLM with all its dependencies
pip install "vllm==0.19.0" --break-system-packages

# Step 2: Overwrite torch with CUDA 12.8 build
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  --break-system-packages

# Step 3: Install extra deps your script needs
pip install "datasets>=3.6.0" "huggingface_hub>=0.30.0" "hf_transfer>=0.1.8" \
  "accelerate>=1.6.0" "sentencepiece>=0.2.0" \
  --break-system-packages

# Step 4: Verify
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
python3 -c "from vllm import LLM, SamplingParams; print('ok')"