# PaddleOCR-VL RunPod Serverless Container (vLLM Backend)
#
# Architecture:
#   paddleocr genai_server (vLLM, port 8080) - VLM inference with continuous batching
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV2 layout detection (CPU, fast)
#     2. Crops → vLLM server at localhost:8080 (batched)
#     3. Post-processing → markdown
#
# Strategy: PaddlePaddle base + install genai_server deps (vLLM) via paddleocr CLI
# Uses separate venv for vLLM server to avoid dependency conflicts

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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PaddleOCR with doc-parser (includes VL model + genai_server CLI)
RUN pip install --ignore-installed "paddleocr[doc-parser]==3.3.3"

# Install genai_server dependencies (vLLM + flash-attn)
# This registers the 'genai_server' subcommand in paddleocr CLI
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
