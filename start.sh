#!/bin/bash
set -e

# Start vLLM-backed VLM server in background
paddleocr genai_server --model_name PaddleOCR-VL-0.9B \
  --host 0.0.0.0 --port 8080 --backend vllm &
VLM_PID=$!

# Wait for vLLM to be healthy
echo "[start.sh] Waiting for vLLM server on port 8080..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "[start.sh] vLLM server ready after ${i}s"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "[start.sh] ERROR: vLLM server failed to start within 120s"
    exit 1
  fi
  sleep 1
done

# Start RunPod handler
python -u /app/handler.py
