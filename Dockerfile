# PaddleOCR-VL-1.5 RunPod Serverless Container (single container, two processes)
#
# Process 1 (base image runtime): PaddleOCR genai_server (vLLM backend), port 8080
# Process 2 (isolated venv): RunPod handler that runs PP-DocLayoutV3/UVDoc on GPU via Paddle
#
# Key design goal: do NOT mix Torch/vLLM and Paddle in the same Python environment.
# That avoids CUDA minor-version ABI conflicts and avoids breaking the prebuilt vLLM stack.

ARG VLM_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu
FROM ${VLM_IMAGE}

USER root
WORKDIR /app

# Minimal system deps used by start.sh (health checks) and common image libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv for PaddleOCRVL pipeline (Paddle + CV models).
RUN python -m venv /opt/paddle_venv && \
    /opt/paddle_venv/bin/python -m pip install --no-cache-dir --upgrade pip

# Determine the base image CUDA version from torch (vLLM stack) and install matching Paddle wheel.
# Production rule: only install Paddle from a *matching* CUDA index. If the base image is built on a
# CUDA minor version that Paddle does not publish, fail the build rather than shipping
# a fragile "closest match" that crashes at runtime.
ARG PADDLE_VERSION=
	RUN CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')" && \
	    echo "Base CUDA_VERSION=${CUDA_VERSION}" && \
	    CUDA_MAJOR="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')" && \
	    case "${CUDA_MAJOR}" in \
	      126) PADDLE_CU="cu126" ;; \
	      128) PADDLE_CU="cu128" ;; \
	      129) PADDLE_CU="cu129" ;; \
	      118) PADDLE_CU="cu118" ;; \
	      *) echo "ERROR: Base CUDA_VERSION=${CUDA_VERSION} is not supported for Paddle GPU wheels. Use a paddleocr-genai-vllm-server image built on CUDA 12.6 (cu126), 12.8 (cu128), or 12.9 (cu129)." >&2; exit 1 ;; \
	    esac && \
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/${PADDLE_CU}/" && \
    if [ -n "${PADDLE_VERSION:-}" ]; then \
      echo "Installing paddlepaddle-gpu==${PADDLE_VERSION} from: ${PADDLE_INDEX}"; \
      /opt/paddle_venv/bin/python -m pip install --no-cache-dir "paddlepaddle-gpu==${PADDLE_VERSION}" -i "${PADDLE_INDEX}"; \
    else \
      echo "Installing latest paddlepaddle-gpu from: ${PADDLE_INDEX}"; \
      /opt/paddle_venv/bin/python -m pip install --no-cache-dir "paddlepaddle-gpu" -i "${PADDLE_INDEX}"; \
    fi

# Install PaddleOCR-VL pipeline + handler deps into the Paddle venv.
# Keep transformers <5 for vLLM compatibility in case the handler imports it indirectly.
RUN BASE_PADDLEOCR_VERSION="$(python -c 'import importlib.metadata as m; print(m.version(\"paddleocr\"))' 2>/dev/null || true)" && \
    if [ -n "${BASE_PADDLEOCR_VERSION}" ]; then \
      echo "Base paddleocr version: ${BASE_PADDLEOCR_VERSION}"; \
      /opt/paddle_venv/bin/python -m pip install --no-cache-dir \
        "paddleocr[doc-parser]==${BASE_PADDLEOCR_VERSION}" \
        "transformers>=4.40.0,<5.0.0" \
        runpod requests; \
    else \
      echo "Base paddleocr version not detected; installing latest compatible paddleocr[doc-parser]."; \
      /opt/paddle_venv/bin/python -m pip install --no-cache-dir \
        "paddleocr[doc-parser]>=3.4.0" \
        "transformers>=4.40.0,<5.0.0" \
        runpod requests; \
    fi

COPY handler.py /app/
COPY --chmod=755 start.sh /app/

ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Baked-in runtime defaults (RunPod env can override)
ENV PADDLE_VL_SERIALIZE=false
ENV CV_DEVICE=gpu
ENV PADDLE_VL_CPU_THREADS=4
ENV PADDLE_VL_MAX_PAGES_PER_BATCH=64
ENV PADDLE_VL_USE_QUEUES=true
ENV PADDLE_VL_VL_REC_MAX_CONCURRENCY=64
ENV PADDLE_VL_DOWNLOAD_WORKERS=20

CMD ["bash", "/app/start.sh"]
