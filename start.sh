#!/bin/bash
set -e

export DISABLE_MODEL_SOURCE_CHECK=True

# Use network volume for model cache (faster cold starts)
VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ -d "$VOLUME_PATH" ]; then
  export HF_HOME="$VOLUME_PATH/huggingface"
  export HF_HUB_CACHE="$VOLUME_PATH/huggingface/hub"
  export PADDLE_HOME="$VOLUME_PATH/paddle_models"
  export PADDLEOCR_HOME="$VOLUME_PATH/paddle_models"
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$PADDLE_HOME"
  echo "[start.sh] Using network volume cache: $VOLUME_PATH"
else
  echo "[start.sh] No network volume found, using container storage"
fi

# Start PaddleOCR genai server with vLLM backend (official method)
# Per: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
echo "[start.sh] Starting PaddleOCR genai_server with vLLM backend..."
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 \
  --port 8080 \
  --backend vllm &
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
