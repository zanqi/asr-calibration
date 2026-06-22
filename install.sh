#!/bin/bash
# Install system-level FFmpeg via conda
conda install -y -c conda-forge ffmpeg

# Install Python packages via pip
pip install -r requirements.txt