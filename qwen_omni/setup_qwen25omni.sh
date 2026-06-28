conda create -n qwen25omni python=3.10 -y
conda activate qwen25omni
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==4.52.3" accelerate
pip install -U qwen-omni-utils
conda install -c conda-forge ffmpeg -y
pip install openai "datasets>=2.18" soundfile librosa huggingface_hub