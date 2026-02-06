# PaddleOCR-VL-1.5 RunPod Serverless Container
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV3 layout detection
#     2. Crops → vLLM server at localhost:8080 (batched)
#     3. UVDoc fallback for collapsed table rows
#     4. Post-processing → markdown
#
# Installation per official docs:
# https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
#
# 1. Install PaddlePaddle GPU
# 2. Install paddleocr[doc-parser]
# 3. Use paddleocr install_genai_server_deps vllm (proper vLLM integration)
# 4. Start with paddleocr genai_server

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

# Install PaddleOCR with doc-parser support
RUN pip install "paddleocr[doc-parser]"

# Install vLLM dependencies via official PaddleOCR method
# This handles proper integration without CUDA conflicts
RUN paddleocr install_genai_server_deps vllm

# Install RunPod SDK
RUN pip install runpod requests

# Pre-download layout model
RUN python -c "from paddleocr import PaddleOCRVL; print('imports ok')" || true

COPY handler.py /app/
COPY --chmod=755 start.sh /app/

ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

CMD ["bash", "/app/start.sh"]
