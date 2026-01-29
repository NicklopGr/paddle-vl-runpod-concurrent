# PaddleOCR-VL RunPod Serverless Container (vLLM Backend)
#
# Architecture:
#   paddleocr genai_server (vLLM, port 8080) - VLM inference with continuous batching
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client:
#     1. PP-DocLayoutV2 layout detection (CPU, fast)
#     2. Crops → vLLM server at localhost:8080 (batched)
#     3. Post-processing → markdown
#
# Base image includes: vLLM + PaddleOCR-VL-0.9B model + paddleocr genai_server

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest

USER root
WORKDIR /app

# Install PaddlePaddle GPU for layout detection (PP-DocLayoutV2)
RUN pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --ignore-installed "paddleocr[doc-parser]>=3.3.2,<3.4"
RUN pip install runpod requests

# Pre-download layout model so cold start doesn't fetch it
RUN python -c "from paddleocr import PaddleOCRVL; print('imports ok')" || true

COPY pipeline_config_vllm.yaml /app/
COPY handler.py /app/
COPY start.sh /app/
RUN chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO

CMD ["bash", "start.sh"]
