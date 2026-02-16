#!/bin/bash
set -euo pipefail

export DISABLE_MODEL_SOURCE_CHECK=True

# ==========================================================================
# H100 GPU CONFIGURATION (RunPod env can override at deploy time)
# ==========================================================================
# - MAX_PAGES_PER_BATCH=64: Process 64 pages simultaneously
# - VL_REC_MAX_CONCURRENCY=64: VLM inference concurrency via vLLM
# - CV_DEVICE=gpu: Layout detection (PP-DocLayoutV3) runs on GPU
# - WORKER_CONCURRENCY=1: Single job at a time for stability
# - CPU_THREADS=4: H100 pods have more CPU cores
: "${PADDLE_VL_WORKER_CONCURRENCY:=1}"
: "${PADDLE_VL_MAX_PAGES_PER_BATCH:=64}"
: "${PADDLE_VL_VL_REC_MAX_CONCURRENCY:=64}"
: "${PADDLE_VL_USE_QUEUES:=true}"
: "${PADDLE_VL_DOWNLOAD_WORKERS:=20}"
: "${PADDLE_VL_SERIALIZE:=false}"
: "${CV_DEVICE:=gpu}"
: "${PADDLE_VL_CPU_THREADS:=4}"
export PADDLE_VL_WORKER_CONCURRENCY PADDLE_VL_MAX_PAGES_PER_BATCH PADDLE_VL_VL_REC_MAX_CONCURRENCY PADDLE_VL_USE_QUEUES PADDLE_VL_DOWNLOAD_WORKERS PADDLE_VL_SERIALIZE CV_DEVICE PADDLE_VL_CPU_THREADS

echo "[start.sh] H100 Config: max_pages=${PADDLE_VL_MAX_PAGES_PER_BATCH}, vl_concurrency=${PADDLE_VL_VL_REC_MAX_CONCURRENCY}, device=${CV_DEVICE}, workers=${PADDLE_VL_WORKER_CONCURRENCY}"

# Limit CPU thread oversubscription (pre/post-processing may still fan out threads under load)
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

# Create vLLM backend config file (--backend_config expects YAML file, not JSON/dict).
# Keep hf-overrides optional; some PaddleOCR builds are strict about JSON-string typing.
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.85}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=16384}"
: "${VLLM_HF_OVERRIDES_JSON:=}" # e.g. {"use_fast": true}

{
  echo "gpu-memory-utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
  echo "max-num-batched-tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
  if [ -n "${VLLM_HF_OVERRIDES_JSON}" ]; then
    # Single-quoted YAML scalar so YAML->argparse gets a *string* and argparse json.loads can parse it.
    echo "hf-overrides: '${VLLM_HF_OVERRIDES_JSON}'"
  fi
} > /tmp/vllm_config.yaml

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

# Start RunPod handler (Paddle runs from /opt/paddle_venv).
#
# Paddle wheels may ship CUDA libs under site-packages/nvidia/*/lib. Prefer them for the handler
# process only (keeps the vLLM process' environment untouched).
echo "[start.sh] Checking Paddle runtime (/opt/paddle_venv)..."
/opt/paddle_venv/bin/python - <<'PY'
import paddle

print("paddle", paddle.__version__, "compiled_with_cuda", paddle.device.is_compiled_with_cuda())
try:
    print("paddle_device", paddle.device.get_device())
except Exception as e:
    print("paddle_device_error", repr(e))
PY

shopt -s nullglob
paddle_lib_dirs=(/opt/paddle_venv/lib/python*/site-packages/nvidia/*/lib)
shopt -u nullglob
if [ "${#paddle_lib_dirs[@]}" -gt 0 ]; then
  PADDLE_NVIDIA_LD_LIBRARY_PATH="$(IFS=:; echo "${paddle_lib_dirs[*]}")"
  echo "[start.sh] Handler LD_LIBRARY_PATH prepend (${#paddle_lib_dirs[@]} dirs)"
  exec env LD_LIBRARY_PATH="${PADDLE_NVIDIA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" /opt/paddle_venv/bin/python -u /app/handler.py
fi

exec /opt/paddle_venv/bin/python -u /app/handler.py
