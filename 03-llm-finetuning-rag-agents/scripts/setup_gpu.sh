#!/usr/bin/env bash
# Setup script for RunPod (RTX PRO 4500, 32 GB VRAM, Ubuntu 22.04).
# Assumes the official RunPod PyTorch template ships with CUDA + torch.
# Run inside the pod (web terminal or SSH).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

echo "=== Routing HF + ChromaDB caches to volume disk ($WORKSPACE) ==="
# Export for this shell (so the rest of the script sees them).
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence-transformers"
export COMET_CACHE="$WORKSPACE/.cache/comet"
export PYTHONUNBUFFERED=1

# Persist for future shells (appended only once).
if ! grep -q "HF_HOME=$WORKSPACE/.cache/huggingface" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

# --- LLM project caches on volume disk ---
export HF_HOME=$WORKSPACE/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
export TRANSFORMERS_CACHE=\$HF_HOME/hub
export SENTENCE_TRANSFORMERS_HOME=\$HF_HOME/sentence-transformers
export COMET_CACHE=$WORKSPACE/.cache/comet
export PYTHONUNBUFFERED=1
EOF
fi
mkdir -p "$HF_HOME/hub" "$SENTENCE_TRANSFORMERS_HOME" "$COMET_CACHE"

echo "=== nvidia-smi check ==="
nvidia-smi

echo "=== Python / torch sanity ==="
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== Apt deps (light) ==="
apt-get update -y
apt-get install -y --no-install-recommends git tmux htop vim libgl1 libglib2.0-0 || true

echo "=== Pip install ==="
pip install --upgrade pip setuptools wheel

# Blackwell (sm_120) needs PyTorch built with CUDA 12.8 wheels.
# RunPod's base image ships torch 2.4.1+cu124 which only goes up to sm_90.
echo "=== Upgrading torch to cu128 (Blackwell-compatible) ==="
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

echo "=== Verifying torch + GPU ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
# Quick CUDA sanity check — small matmul on GPU
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
z = x @ y
torch.cuda.synchronize()
print("CUDA matmul OK, result norm:", float(z.norm()))
PY

echo "=== Installing remaining deps ==="
pip install -r requirements.txt

echo "=== HF login (if HF_TOKEN set in environment) ==="
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "=== Done. Next steps:"
echo "  1. bash scripts/download_data.sh"
echo "  2. python -m src.task1_lora_mt.train --config configs/lora_config.yaml"
df -h "$WORKSPACE" /
