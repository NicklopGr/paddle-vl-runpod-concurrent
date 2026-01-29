#!/bin/bash
set -e

export VLLM_DISABLE_MODEL_SOURCE_CHECK=1

# Start vLLM server directly (serves PaddleOCR-VL via OpenAI-compatible API)
# Per: https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html
vllm serve PaddlePaddle/PaddleOCR-VL \
  --host 0.0.0.0 \
  --port 8080 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --mm-processor-kwargs '{"use_fast": true}' &
VLM_PID=$!

# Wait for vLLM to be healthy
echo "[start.sh] Waiting for vLLM server on port 8080..."
for i in $(seq 1 180); do
  if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "[start.sh] vLLM server ready after ${i}s"
    break
  fi
  if [ "$i" -eq 180 ]; then
    echo "[start.sh] ERROR: vLLM server failed to start within 180s"
    exit 1
  fi
  sleep 1
done

# Start RunPod handler
python -u /app/handler.py
