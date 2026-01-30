# PaddleOCR-VL RunPod Serverless Container (Concurrent Version)
#
# Same as paddle-vl-runpod but with concurrency_modifier for in-handler concurrency.
# Multiple jobs can run simultaneously on the same GPU worker.
#
# Uses official PaddlePaddle 3.2.2 image with CUDA 12.6 + cuDNN 9.5
# PaddleOCR 3.3.3 (latest as of Jan 26, 2026)
#
# Hardware Requirements:
# - GPU with compute capability >= 7.0 (Volta or newer)
# - Recommended: Ampere+ (RTX 30xx, A10G, A100, etc.)
# - VRAM: ~3-4 GB with Flash Attention, ~40 GB without
#
# GPU Driver Requirements: >= 550.54.14 (Linux)

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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PaddleOCR 3.3.3 with doc-parser (includes VL model)
RUN pip install --ignore-installed "paddleocr[doc-parser]==3.3.3"

# Install RunPod SDK and requests (for URL downloading)
RUN pip install runpod requests

# Copy handler and warmup scripts
COPY handler.py warmup.py ./

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1

# Skip model source connectivity checks (~12s savings on cold start)
ENV DISABLE_MODEL_SOURCE_CHECK=True

# Persist vLLM torch.compile cache on network volume (~32s savings on subsequent cold starts)
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm_cache

# Default: serialize pipeline.predict() calls (safe default)
# Set to "false" if you confirm pipeline is thread-safe
ENV PADDLE_VL_SERIALIZE=true

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
