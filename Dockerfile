# PaddleOCR-VL-1.5 RunPod Serverless Container
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV3 layout detection
#     2. Crops → vLLM server at localhost:8080 (batched)
#     3. UVDoc fallback for collapsed table rows
#     4. Post-processing → markdown
#
# Base image: Official PaddleOCR genai vLLM server
# - All dependencies pre-installed and compatible
# - No CUDA version conflicts
# - Proper paddleocr + vLLM integration
#
# Per: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu

WORKDIR /app

# Install RunPod SDK
RUN pip install runpod requests

# Copy handler and startup script (start.sh must be executable in repo)
COPY handler.py /app/
COPY --chmod=755 start.sh /app/

ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True

CMD ["bash", "/app/start.sh"]
