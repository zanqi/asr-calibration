#!/usr/bin/env bash
# Setup for evaluating LLaMA-Omni2-0.5B on the EAR benchmark.
# Uses the official LLaMA-Omni2 repo, in its OWN conda env (keep it separate
# from qwen3omni / mini-omni2 — their pins conflict).
set -e

ENV_NAME=llama-omni2

conda create -n "$ENV_NAME" python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 1. Clone and install the repo (its pyproject pulls torch/transformers/whisper/etc.)
git clone https://github.com/ictnlp/LLaMA-Omni2
cd LLaMA-Omni2
pip install -e .

# 2. Speech encoder: Whisper-large-v3 (~3 GB) -> models/speech_encoder/
python -c "import whisper; whisper.load_model('large-v3', download_root='models/speech_encoder/')"

# 3. LLaMA-Omni2-0.5B weights (English).
#    The CosyVoice2 decoder is only needed for SPEECH output; we score TEXT only,
#    so it's left commented out. Uncomment if you also want spoken responses.
model_name=LLaMA-Omni2-0.5B
huggingface-cli download --resume-download ICTNLP/$model_name --local-dir models/$model_name
# huggingface-cli download --resume-download ICTNLP/cosy2_decoder --local-dir models/cosy2_decoder

# 4. Eval extras: GPT-4o judge + dataset + audio decode (soundfile, no torchcodec).
conda install -c conda-forge ffmpeg -y
pip install openai "datasets>=2.18" soundfile librosa huggingface_hub

echo
echo "Done.  conda activate $ENV_NAME"
echo "Copy ear_eval.py into this LLaMA-Omni2/ directory, then:"
echo "  export OPENAI_API_KEY=...     # and HF_TOKEN (dataset is private)"
echo "  python ear_eval.py --num-samples 3   # smoke test"