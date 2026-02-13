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

# Base image (paddlex-genai-vllm-server) has vLLM pre-installed with specific torch version.
# We need to preserve the torch/torchvision compatibility after installing paddleocr.

# Step 1: Save base image's torch version and index URL before paddleocr potentially breaks it
RUN python -c "import torch; v=torch.__version__; print(f'TORCH_VERSION={v}')" > /tmp/base_versions.env && \
    python -c "import torch; cuda=torch.version.cuda; print(f'CUDA_VERSION={cuda}')" >> /tmp/base_versions.env && \
    cat /tmp/base_versions.env

# Step 2: Install PaddleOCR (this may install incompatible torch/torchvision versions)
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0"

# Step 3: Fix torch/torchvision/transformers compatibility
#
# Problem: paddleocr[doc-parser] installs incompatible package versions that break vLLM:
#   1. torchvision gets upgraded to version incompatible with base image's torch
#      → RuntimeError: operator torchvision::nms does not exist
#   2. transformers gets upgraded to 5.0+ which breaks vLLM's ProcessorMixin import
#      → ModuleNotFoundError: Could not import module 'ProcessorMixin'
#
# Solution: Restore torch/torchvision to base image versions, fix transformers
#
# References:
#   - https://github.com/PaddlePaddle/PaddleOCR/issues/16823
#   - https://github.com/vllm-project/vllm/issues/18776
#   - https://github.com/pytorch/vision#installation
RUN . /tmp/base_versions.env && \
    echo "Restoring torch=${TORCH_VERSION} for CUDA ${CUDA_VERSION}" && \
    # Determine the correct PyTorch wheel index based on CUDA version
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') && \
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_MAJOR}" && \
    echo "Using PyTorch index: ${PYTORCH_INDEX}" && \
    # Extract torch version without build suffix (e.g., 2.8.0+cu126 -> 2.8.0)
    TORCH_BASE=$(echo $TORCH_VERSION | cut -d'+' -f1) && \
    # Determine matching torchvision version based on torch version
    # Compatibility: torch 2.5->tv0.20, 2.6->0.21, 2.7->0.22, 2.8->0.23, 2.9->0.24, 2.10->0.25
    TORCH_MINOR=$(echo $TORCH_BASE | cut -d. -f2) && \
    TV_MINOR=$((TORCH_MINOR + 15)) && \
    TV_VERSION="0.${TV_MINOR}.0" && \
    echo "Torch ${TORCH_BASE} requires torchvision ${TV_VERSION}" && \
    # Reinstall torch and torchvision from the correct index
    pip install --no-cache-dir \
        "torch==${TORCH_BASE}" \
        "torchvision==${TV_VERSION}" \
        "transformers>=4.40.0,<5.0.0" \
        --index-url "${PYTORCH_INDEX}" && \
    # Verify installation
    python -c "import torch; import torchvision; print(f'torch={torch.__version__}, torchvision={torchvision.__version__}')"

# Install RunPod SDK
RUN pip install --no-cache-dir runpod requests

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
