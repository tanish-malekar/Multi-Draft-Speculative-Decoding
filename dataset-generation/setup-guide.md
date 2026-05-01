# RunPod + VS Code Setup Guide
## Distillation Dataset Generation on A100-80GB

---

## STEP 1: Create RunPod Pod

1. Go to https://runpod.io → **Pods** → **+ GPU Pod**
2. Select GPU: **A100 80GB SXM** (~$1.64/hr on Community Cloud)
   - If unavailable, fallback: **A100 80GB PCIe** (~$1.44/hr)
   - Do NOT pick Blackwell/RTX Pro 6000 (driver compat issues)
3. Template: **RunPod PyTorch 2.1** (CUDA 12.1)
4. Disk: **Container 50GB** + **Volume 100GB** (volume persists across restarts)
5. Check **Expose SSH (port 22)**
6. Click **Deploy**

Wait ~2 min for pod to start. Note the pod ID.

---

## STEP 2: Connect VS Code via SSH

### 2a. Get SSH connection info
- Go to your pod → **Connect** → copy the SSH command
- It will look like: `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`

### 2b. Add to VS Code
1. Install VS Code extension: **Remote - SSH** (by Microsoft)
2. Press `Ctrl+Shift+P` → **Remote-SSH: Add New SSH Host**
3. Paste the SSH command from RunPod
4. Select config file (usually `~/.ssh/config`)
5. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → select your pod

### 2c. If you don't have an SSH key on RunPod
- In RunPod dashboard → **Settings** → **SSH Public Keys**
- Add your public key (`cat ~/.ssh/id_ed25519.pub` or `cat ~/.ssh/id_rsa.pub`)
- If you don't have one: `ssh-keygen -t ed25519` on your local machine

---

## STEP 3: Setup Environment on Pod

Once VS Code is connected to the pod, open a terminal (`Ctrl+``):

```bash
# Verify GPU
nvidia-smi
# Should show: A100-SXM4-80GB

# Verify CUDA
nvcc --version
# Should show: 12.1.x

# Go to persistent volume (survives pod restarts)
cd /workspace

# Clone your project or create directory
mkdir -p distillation && cd distillation

# Install dependencies
pip install vllm==0.6.3 datasets transformers accelerate sentencepiece protobuf --break-system-packages

# Verify vLLM works
python -c "from vllm import LLM; print('vLLM OK')"
```

---

## STEP 4: Upload Script

In VS Code (connected to pod), create these files in `/workspace/distillation/`:

1. Copy `generate_distillation_data.py` into this directory
2. Copy `requirements.txt` into this directory

Or from local machine:
```bash
scp -P <port> generate_distillation_data.py root@<ip>:/workspace/distillation/
```

---

## STEP 5: Run — Test First

```bash
cd /workspace/distillation

# Test with 10 samples per domain to verify everything works
python generate_distillation_data.py \
    --samples_per_domain 10 \
    --output_dir ./test_data \
    --checkpoint_every 5

# Check output
ls -la ./test_data/
cat ./test_data/math_distillation.jsonl | head -2 | python -m json.tool
```

---

## STEP 6: Run — Full Generation

```bash
# Use tmux so it survives SSH disconnections!
tmux new -s distill

cd /workspace/distillation

# Full run — 50K samples per domain
python generate_distillation_data.py \
    --samples_per_domain 50000 \
    --batch_size 64 \
    --checkpoint_every 500 \
    --output_dir ./distillation_data

# Detach from tmux: Ctrl+B then D
# Re-attach later:  tmux attach -t distill
```

### Why tmux?
Your SSH connection WILL drop at some point during a 10+ hour job.
Without tmux, the process dies. With tmux, it keeps running.

---

## STEP 7: If It Crashes — Resume

Just re-run the exact same command:

```bash
# This will auto-detect the checkpoint and resume from where it stopped
python generate_distillation_data.py \
    --samples_per_domain 50000 \
    --batch_size 64 \
    --checkpoint_every 500 \
    --output_dir ./distillation_data
```

The script will print something like:
```
    ↻ RESUMING from sample 23,500/57,500 (40.9% already done)
```

### To start over instead of resuming:
```bash
python generate_distillation_data.py \
    --samples_per_domain 50000 \
    --output_dir ./distillation_data \
    --fresh
```

---

## STEP 8: Download Results

### Option A: scp from local machine
```bash
scp -P <port> -r root@<ip>:/workspace/distillation/distillation_data/ ./
```

### Option B: RunPod volume (persists across pod restarts)
Data in `/workspace/` persists as long as your volume exists.
You can stop the pod (stop billing) and restart later to download.

### Option C: Push to HuggingFace Hub
```bash
pip install huggingface_hub --break-system-packages
huggingface-cli login

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./distillation_data',
    repo_id='your-username/qwen-distillation-data',
    repo_type='dataset',
)
"
```

---

## STEP 9: Stop Pod (stop billing!)

⚠️ RunPod charges per hour even when idle.

```bash
# In RunPod dashboard: click STOP on your pod
# Your /workspace volume is preserved
# You only pay volume storage (~$0.10/GB/month)
```

---

## EXPECTED TIMELINE & COST

| Phase           | Time        | Cost     |
|-----------------|-------------|----------|
| Setup + test    | 15 min      | ~$0.50   |
| Math (57.5K)    | ~3-4 hrs    | ~$5.50   |
| Code (57.5K)    | ~5-6 hrs    | ~$9.00   |
| Translation     | ~1-2 hrs    | ~$2.50   |
| **Total**       | **~10-13 hrs** | **~$18-20** |

Code takes longest because max_tokens=1024 (vs 512 for others).

---

## TROUBLESHOOTING

**"CUDA out of memory"**
→ Reduce batch_size: `--batch_size 32`

**"Model not found"**
→ May need HuggingFace login: `huggingface-cli login`
→ Qwen2.5-7B-Instruct is gated, accept terms on HF first

**vLLM hangs on model loading**
→ First load downloads ~14GB of weights. Be patient (~5 min).
→ Subsequent runs use cached weights from `/root/.cache/huggingface/`

**SSH disconnects during run**
→ This is why we use tmux. Re-attach: `tmux attach -t distill`
→ The script checkpoints, so even without tmux, re-run resumes.

**"Killed" (OOM at OS level)**
→ Reduce gpu_memory_utilization in script from 0.85 to 0.80
→ Or reduce batch_size to 32

**Dataset download fails**
→ Some datasets need `huggingface-cli login`
→ Run `--domains math` first, then code, then translation separately