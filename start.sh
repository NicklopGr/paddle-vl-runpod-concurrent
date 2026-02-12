#!/bin/bash
set -e

export DISABLE_MODEL_SOURCE_CHECK=True

# Baked-in defaults (RunPod env can override these at deploy time)
: "${PADDLE_VL_SERIALIZE:=false}"
: "${CV_DEVICE:=cpu}"
: "${PADDLE_VL_USE_QUEUES:=true}"
: "${PADDLE_VL_VL_REC_MAX_CONCURRENCY:=20}"
: "${PADDLE_VL_MAX_PAGES_PER_BATCH:=9}"
: "${PADDLE_VL_DOWNLOAD_WORKERS:=20}"
: "${PADDLE_VL_CPU_THREADS:=1}"
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
  # PaddleX uses multiple env vars for model caching:
  # - PADDLEX_HOME: base directory (often ignored for official models)
  # - HUB_HOME / PADDLE_HUB_HOME: where "official_models" are actually downloaded
  export PADDLEX_HOME="$VOLUME_PATH/paddlex_models"
  export HUB_HOME="$VOLUME_PATH/paddlex_models/official_models"
  export PADDLE_HUB_HOME="$VOLUME_PATH/paddlex_models/official_models"
  export HF_HOME="$VOLUME_PATH/huggingface"
  export HF_HUB_CACHE="$VOLUME_PATH/huggingface/hub"
  mkdir -p "$PADDLEX_HOME" "$HUB_HOME" "$HF_HOME" "$HF_HUB_CACHE"
  echo "[start.sh] Using network volume cache: $VOLUME_PATH"
  echo "[start.sh] PADDLEX_HOME=$PADDLEX_HOME"
  echo "[start.sh] HUB_HOME=$HUB_HOME"
else
  echo "[start.sh] No network volume found, using container storage"
fi

# Create vLLM backend config file (--backend_config expects YAML file, not string)
# Note: hf-overrides enables fast image processor (requires torchvision)
cat > /tmp/vllm_config.yaml << 'EOF'
gpu-memory-utilization: 0.5
hf-overrides:
  use_fast: true
EOF

# Start PaddleOCR genai server with vLLM backend
echo "[start.sh] Starting PaddleOCR genai_server with vLLM backend (gpu-memory-utilization=0.5)..."
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
