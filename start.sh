#!/bin/bash
set -e

export DISABLE_MODEL_SOURCE_CHECK=True

# Baked-in defaults optimized for H100 GPU (RunPod env can override these at deploy time)
# H100 optimization per Baidu official PaddleOCR-VL-1.5 config:
# - batch_size: 64 (official pipeline batch size)
# - CV_DEVICE=gpu (uses paddlepaddle-gpu from base image for layout detection)
# - CPU_THREADS=4 (H100 pods have more CPU cores)
# NOTE: Both VLM inference AND layout detection (PP-DocLayoutV3) run on GPU
: "${PADDLE_VL_SERIALIZE:=false}"
: "${CV_DEVICE:=gpu}"
: "${PADDLE_VL_USE_QUEUES:=true}"
: "${PADDLE_VL_VL_REC_MAX_CONCURRENCY:=64}"
: "${PADDLE_VL_MAX_PAGES_PER_BATCH:=64}"
: "${PADDLE_VL_DOWNLOAD_WORKERS:=20}"
: "${PADDLE_VL_CPU_THREADS:=4}"
export PADDLE_VL_SERIALIZE CV_DEVICE PADDLE_VL_USE_QUEUES PADDLE_VL_VL_REC_MAX_CONCURRENCY PADDLE_VL_MAX_PAGES_PER_BATCH PADDLE_VL_DOWNLOAD_WORKERS PADDLE_VL_CPU_THREADS

echo "[start.sh] Settings: PADDLE_VL_SERIALIZE=${PADDLE_VL_SERIALIZE}, CV_DEVICE=${CV_DEVICE}, PADDLE_VL_CPU_THREADS=${PADDLE_VL_CPU_THREADS}, PADDLE_VL_MAX_PAGES_PER_BATCH=${PADDLE_VL_MAX_PAGES_PER_BATCH}, PADDLE_VL_USE_QUEUES=${PADDLE_VL_USE_QUEUES}, PADDLE_VL_VL_REC_MAX_CONCURRENCY=${PADDLE_VL_VL_REC_MAX_CONCURRENCY}, PADDLE_VL_DOWNLOAD_WORKERS=${PADDLE_VL_DOWNLOAD_WORKERS}"

# Limit CPU thread oversubscription (layout runs on CPU; many concurrent requests can thrash without caps)
CPU_THREADS="${PADDLE_VL_CPU_THREADS}"
: "${OMP_NUM_THREADS:=$CPU_THREADS}"
: "${MKL_NUM_THREADS:=$CPU_THREADS}"
: "${OPENBLAS_NUM_THREADS:=$CPU_THREADS}"
: "${NUMEXPR_NUM_THREADS:=$CPU_THREADS}"
: "${VECLIB_MAXIMUM_THREADS:=$CPU_THREADS}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS

echo "[start.sh] CPU thread caps: OMP_NUM_THREADS=${OMP_NUM_THREADS}, MKL_NUM_THREADS=${MKL_NUM_THREADS}, OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"

# Use network volume for model cache (faster cold starts)
VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ -d "$VOLUME_PATH" ]; then
  # Paddle runtime paths observed in logs are under /root/.paddlex and /root/.cache.
  # Bind those paths onto the volume so downloads/compile cache persist across restarts.
  export PADDLEX_HOME="$VOLUME_PATH/paddlex_models"
  export HUB_HOME="$VOLUME_PATH/paddlex_models/official_models"
  export PADDLE_HUB_HOME="$HUB_HOME"
  export HF_HOME="$VOLUME_PATH/huggingface"
  export HF_HUB_CACHE="$VOLUME_PATH/huggingface/hub"
  export XDG_CACHE_HOME="$VOLUME_PATH/xdg_cache"
  export TORCH_HOME="$VOLUME_PATH/torch_cache"
  mkdir -p "$PADDLEX_HOME" "$HUB_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME/vllm" "$TORCH_HOME"

  mkdir -p /root/.paddlex /root/.cache
  rm -rf /root/.paddlex/official_models /root/.cache/huggingface /root/.cache/vllm /root/.cache/torch
  ln -s "$HUB_HOME" /root/.paddlex/official_models
  ln -s "$HF_HUB_CACHE" /root/.cache/huggingface
  ln -s "$XDG_CACHE_HOME/vllm" /root/.cache/vllm
  ln -s "$TORCH_HOME" /root/.cache/torch

  echo "[start.sh] Using network volume cache: $VOLUME_PATH"
  echo "[start.sh] PADDLEX_HOME=$PADDLEX_HOME"
  echo "[start.sh] HUB_HOME=$HUB_HOME"
  echo "[start.sh] /root/.paddlex/official_models -> $(readlink -f /root/.paddlex/official_models || true)"
  echo "[start.sh] /root/.cache/vllm -> $(readlink -f /root/.cache/vllm || true)"
else
  echo "[start.sh] No network volume found, using container storage"
fi

# Runtime defaults for stable single-worker behavior (H100 optimized)
: "${PADDLE_VL_WORKER_CONCURRENCY:=1}"
: "${PADDLE_VL_MAX_PAGES_PER_BATCH:=64}"
: "${PADDLE_VL_VL_REC_MAX_CONCURRENCY:=64}"
: "${PADDLE_VL_USE_QUEUES:=true}"
: "${PADDLE_VL_DOWNLOAD_WORKERS:=20}"
export PADDLE_VL_WORKER_CONCURRENCY PADDLE_VL_MAX_PAGES_PER_BATCH PADDLE_VL_VL_REC_MAX_CONCURRENCY PADDLE_VL_USE_QUEUES PADDLE_VL_DOWNLOAD_WORKERS
echo "[start.sh] Runtime: worker_concurrency=${PADDLE_VL_WORKER_CONCURRENCY}, max_pages_per_batch=${PADDLE_VL_MAX_PAGES_PER_BATCH}, vl_rec_max_concurrency=${PADDLE_VL_VL_REC_MAX_CONCURRENCY}, use_queues=${PADDLE_VL_USE_QUEUES}, download_workers=${PADDLE_VL_DOWNLOAD_WORKERS}"

# Create vLLM backend config file (--backend_config expects YAML file, not string)
# H100 optimized: 85% GPU memory utilization, increased batch tokens
# Note: hf-overrides enables fast image processor (requires torchvision)
cat > /tmp/vllm_config.yaml << 'EOF'
gpu-memory-utilization: 0.85
max-num-batched-tokens: 16384
hf-overrides: "{\"use_fast\": true}"
EOF

# Start PaddleOCR genai server with vLLM backend
echo "[start.sh] Starting PaddleOCR genai_server with vLLM backend (gpu-memory-utilization=0.85, max-num-batched-tokens=16384)..."
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 \
  --port 8080 \
  --backend vllm \
  --backend_config /tmp/vllm_config.yaml &
VLM_PID=$!

# Wait for server to be healthy
echo "[start.sh] Waiting for genai_server on port 8080..."
for i in $(seq 1 300); do
  if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "[start.sh] genai_server ready after ${i}s"
    break
  fi
  if [ "$i" -eq 300 ]; then
    echo "[start.sh] ERROR: genai_server failed to start within 300s"
    exit 1
  fi
  sleep 1
done

# Start RunPod handler
python -u /app/handler.py
