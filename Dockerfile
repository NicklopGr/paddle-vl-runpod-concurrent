# PaddleOCR-VL RunPod Serverless Container (vLLM Backend)
#
# Architecture:
#   paddleocr genai_server (vLLM, port 8080) - VLM inference with continuous batching
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV2 layout detection (CPU, fast)
#     2. Crops → vLLM server at localhost:8080 (batched)
#     3. Post-processing → markdown
#
# Strategy: PaddlePaddle base + prebuilt flash-attn wheel + paddleocr genai_server deps
# flash-attn must be installed BEFORE install_genai_server_deps to avoid nvcc compilation

FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PaddleOCR with doc-parser (includes VL model + genai_server CLI)
RUN pip install --ignore-installed "paddleocr[doc-parser]==3.3.3"

# Install prebuilt flash-attn wheel FIRST (avoids nvcc compilation during build)
# Python 3.10, CUDA 12.4 (forward-compatible with 12.6 runtime), torch 2.8
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2%2Bcu124torch2.8-cp310-cp310-linux_x86_64.whl

# git is required by vLLM's pip install (separate layer to bust cache)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Now install genai_server dependencies (vLLM) - flash-attn already satisfied
RUN paddleocr install_genai_server_deps vllm

# Install RunPod SDK
RUN pip install runpod requests

# Pre-download layout model so cold start doesn't fetch it
RUN python -c "from paddleocr import PaddleOCRVL; print('imports ok')" || true

COPY pipeline_config_vllm.yaml /app/
COPY handler.py /app/
COPY start.sh /app/
RUN chmod +x /app/start.sh

ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True

CMD ["bash", "start.sh"]
