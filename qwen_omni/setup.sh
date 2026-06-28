#!/usr/bin/env bash
# Setup for running ear_eval.py on an A100.
# Pinning Python 3.10 for smoother install
set -e

ENV_NAME=qwen3omni

# 1. Fresh env on Python 3.10 (NOT 3.13)
conda create -n "$ENV_NAME" python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2. PyTorch with CUDA (cu124 works well on A100). Install torch FIRST so that
#    flash-attn can build/select against it.
pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu130

# 3. Transformers from source (Qwen3-Omni support) + accelerate
pip install "git+https://github.com/huggingface/transformers"
pip install accelerate

# 4. Qwen omni helpers (needs ffmpeg on the system)
pip install -U qwen-omni-utils
conda install -c conda-forge ffmpeg -y   # or: sudo apt-get install -y ffmpeg

# 5. Audio + dataset + judge deps
pip install soundfile librosa "datasets>=2.18" huggingface_hub anthropic

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Run the eval with sdpa attention:  python ear_eval.py --attn sdpa --num-samples 50"