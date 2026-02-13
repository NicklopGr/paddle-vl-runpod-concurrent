# PaddleOCR-VL-1.5 RunPod Serverless Container
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV3 layout detection
#     2. Crops -> vLLM server at localhost:8080 (batched)
#     3. UVDoc fallback for collapsed table rows
#     4. Post-processing -> markdown
#
# Uses official pre-built base image that has vLLM with compatible CUDA libraries.
# Per: https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/paddleocr_vl_docker
#
# This avoids the nvidia-cublas version conflict between paddlepaddle-gpu and vllm.

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest

USER root

WORKDIR /app

# Install system dependencies for OpenCV and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1-mesa-glx     libglib2.0-0     libsm6     libxext6     libxrender-dev     libgomp1     poppler-utils     curl     && rm -rf /var/lib/apt/lists/*

# Base image (paddlex-genai-vllm-server) has paddlepaddle-gpu 3.0.0b2 pre-installed
# Using GPU paddle allows CV_DEVICE=gpu for layout detection (PP-DocLayoutV3)
# This eliminates "cv worker: std::exception" crashes from GPU/CPU state mismatch

# Install PaddleOCR with doc-parser support
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0"

# Fix transformers version for vLLM 0.10.2 compatibility (base image requirement)
# Problem: paddleocr[doc-parser] pulls in transformers >= 5.0.0 which breaks vLLM
# - vLLM 0.10.2 requires transformers < 5.0.0 (ProcessorMixin import fails otherwise)
# - PaddleX genai-vllm-server specifies: transformers < 5.0.0
# - PaddleOCR FAQ #16823 recommends: transformers==4.57.6
# See: https://github.com/PaddlePaddle/PaddleOCR/issues/16823
RUN pip install --no-cache-dir "transformers==4.57.6"

# Install RunPod SDK
RUN pip install --no-cache-dir runpod requests

# Install FlashInfer for optimal vLLM sampling (H100 optimized)
# Using cu124 (CUDA 12.4) for compatibility with base image
RUN pip install --no-cache-dir flashinfer-python==0.2.2+cu124torch2.5 \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5/

# Pre-download layout model (PP-DocLayoutV3)
RUN python -c "from paddleocr import PaddleOCRVL; print('PaddleOCR-VL imports ok')" || true

COPY handler.py /app/
COPY --chmod=755 start.sh /app/

ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Baked-in runtime defaults optimized for H100 GPU (can be overridden via RunPod env)
# H100 optimization per Baidu official PaddleOCR-VL-1.5 config
ENV PADDLE_VL_SERIALIZE=false
# CV_DEVICE=gpu uses paddlepaddle-gpu for layout detection (PP-DocLayoutV3)
# Base image has paddlepaddle-gpu 3.0.0b2 with compatible CUDA libs
ENV CV_DEVICE=gpu
ENV PADDLE_VL_CPU_THREADS=4
ENV PADDLE_VL_MAX_PAGES_PER_BATCH=64
ENV PADDLE_VL_USE_QUEUES=true
ENV PADDLE_VL_VL_REC_MAX_CONCURRENCY=64
ENV PADDLE_VL_DOWNLOAD_WORKERS=20

CMD ["bash", "/app/start.sh"]
