"""
PaddleOCR-VL RunPod Serverless Handler (vLLM Backend)

Architecture:
  - paddleocr genai_server runs vLLM with PaddleOCR-VL-0.9B (continuous batching)
  - This handler uses PaddleOCRVL pipeline client which:
    1. Runs PP-DocLayoutV2 layout detection (CPU)
    2. Sends cropped regions to vLLM server at localhost:8080
    3. Assembles markdown output

Input (URL mode - preferred):
{
    "input": {
        "image_urls": ["https://...", ...],
        "skip_resize": false,
        "warmup": true
    }
}

Input (base64 mode - backward compatible):
{
    "input": {
        "images_base64": ["base64_1", ...],
        "image_base64": "base64_single"
    }
}

Output: { status, result: { pages: [{page_number, markdown, parsing_res_list}], ocrProvider, processingTime } }
"""

import runpod
import asyncio
import base64
import tempfile
import os
import time
import warnings
import requests
from PIL import Image
import io
import numpy as np

# ============================================================================
# PERFORMANCE OPTIMIZATIONS (vLLM backend v2)
# ============================================================================

os.environ["PADDLEX_SKIP_MODEL_CHECK"] = "1"
warnings.filterwarnings("ignore", message=".*Non compatible API.*")
warnings.filterwarnings("ignore", category=Warning, module="paddle.utils.decorator_utils")

# Global pipeline - loaded once
paddle_vl_pipeline = None

# Serialization lock for pipeline.predict()
_predict_lock: asyncio.Lock | None = None
SERIALIZE_PREDICT = os.environ.get("PADDLE_VL_SERIALIZE", "true").lower() == "true"


def load_pipeline():
    """Load PaddleOCR-VL pipeline with vLLM backend config"""
    global paddle_vl_pipeline
    if paddle_vl_pipeline is not None:
        return paddle_vl_pipeline

    print("[PaddleOCR-VL] Loading pipeline with vLLM backend...")
    start = time.time()

    from paddleocr import PaddleOCRVL
    paddle_vl_pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://localhost:8080/v1",
    )

    elapsed = time.time() - start
    print(f"[PaddleOCR-VL] Pipeline loaded in {elapsed:.2f}s (vLLM backend)")
    return paddle_vl_pipeline


def resize_image_if_needed(image: Image.Image, max_dimension: int = 1920) -> Image.Image:
    """Resize image if it exceeds max dimension while preserving aspect ratio."""
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return image

    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    print(f"[PaddleOCR-VL] Resizing image from {width}x{height} to {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-safe types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def download_image(url: str) -> bytes:
    """Download image from URL and return raw bytes"""
    print(f"[PaddleOCR-VL] Downloading image from URL ({len(url)} chars)...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    print(f"[PaddleOCR-VL] Downloaded {len(resp.content)} bytes")
    return resp.content


def process_single_image(pipeline, image_bytes: bytes, page_number: int, skip_resize: bool = False) -> dict:
    """Process a single image and return markdown + structured output"""
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if skip_resize:
        print(f"[PaddleOCR-VL] Skipping resize, using original {image.size[0]}x{image.size[1]}")
    else:
        image = resize_image_if_needed(image, max_dimension=1920)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name, 'PNG')
        tmp_path = tmp.name

    try:
        results = pipeline.predict(tmp_path)

        markdown_output = ""
        parsing_res_list = []
        json_output = None

        for res in results:
            try:
                md_info = res.markdown
                if md_info:
                    if isinstance(md_info, dict):
                        md_texts = md_info.get('markdown_texts', '')
                        if isinstance(md_texts, str):
                            markdown_output += md_texts
                        elif isinstance(md_texts, list):
                            markdown_output += '\n\n'.join(str(t) for t in md_texts)
                    elif isinstance(md_info, str):
                        markdown_output += md_info
            except Exception as e:
                print(f"[PaddleOCR-VL] Error accessing markdown: {e}")

            try:
                json_data = res.json
                if json_data:
                    json_output = convert_to_serializable(json_data)
                    if isinstance(json_data, dict) and 'parsing_res_list' in json_data:
                        parsing_res_list = convert_to_serializable(json_data['parsing_res_list'])
            except Exception as e:
                print(f"[PaddleOCR-VL] Error accessing json: {e}")

            if not markdown_output and parsing_res_list:
                for block in parsing_res_list:
                    label = block.get('block_label', '')
                    content = block.get('block_content', '')
                    if content:
                        if label == 'table':
                            markdown_output += f"\n\n{content}\n\n"
                        else:
                            markdown_output += f"\n{content}\n"

        return {
            "page_number": page_number,
            "markdown": markdown_output.strip(),
            "parsing_res_list": parsing_res_list,
            "json": json_output
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def process_single_image_async(pipeline, image_bytes: bytes, page_number: int, skip_resize: bool = False) -> dict:
    """Async wrapper with optional serialization lock"""
    global _predict_lock

    if SERIALIZE_PREDICT:
        if _predict_lock is None:
            _predict_lock = asyncio.Lock()
        async with _predict_lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, process_single_image, pipeline, image_bytes, page_number, skip_resize
            )
    else:
        return await asyncio.get_event_loop().run_in_executor(
            None, process_single_image, pipeline, image_bytes, page_number, skip_resize
        )


async def handler(event):
    """RunPod serverless handler (async, concurrent-capable)"""
    start_time = time.time()

    try:
        job_input = event.get("input", {}) or {}

        # Warmup path
        if event.get("warmup") or job_input.get("warmup"):
            load_pipeline()
            return {
                "status": "success",
                "result": {
                    "warmup": True,
                    "backend": "vllm",
                    "concurrent": True,
                    "serialize_predict": SERIALIZE_PREDICT
                }
            }

        skip_resize = job_input.get("skip_resize", False)

        # Collect image bytes from URLs or base64
        image_bytes_list: list[bytes] = []

        # Priority 1: image_urls (preferred)
        image_urls = job_input.get("image_urls", [])
        if image_urls:
            print(f"[PaddleOCR-VL] Downloading {len(image_urls)} images from URLs...")
            for url in image_urls:
                image_bytes_list.append(download_image(url))

        # Priority 2: images_base64 (backward compatible)
        if not image_bytes_list:
            images_base64 = job_input.get("images_base64", [])
            if not images_base64 and job_input.get("image_base64"):
                images_base64 = [job_input.get("image_base64")]

            for b64 in images_base64:
                image_bytes_list.append(base64.b64decode(b64))

        if not image_bytes_list:
            return {
                "status": "error",
                "error": "No images provided. Send 'image_urls', 'images_base64', or 'image_base64'."
            }

        if skip_resize:
            print(f"[PaddleOCR-VL] skip_resize=True (client handled sizing)")

        print(f"[PaddleOCR-VL] Processing {len(image_bytes_list)} page(s) (vLLM backend)")

        pipeline = load_pipeline()

        pages = []
        for i, img_bytes in enumerate(image_bytes_list):
            page_start = time.time()
            page_result = await process_single_image_async(
                pipeline, img_bytes, page_number=i + 1, skip_resize=skip_resize
            )
            page_time = time.time() - page_start
            print(f"[PaddleOCR-VL] Page {i + 1} processed in {page_time:.2f}s")
            pages.append(page_result)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "result": {
                "pages": pages,
                "ocrProvider": "paddleocr-vl-vllm",
                "processingTime": processing_time
            }
        }

    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"[PaddleOCR-VL] Error: {error_msg}\n{stack_trace}")
        return {
            "status": "error",
            "error": error_msg,
            "stack_trace": stack_trace
        }


def concurrency_modifier(current_concurrency: int) -> int:
    """
    Allow up to 20 concurrent jobs.
    vLLM handles batching internally, so concurrent requests are efficient.
    """
    return 20


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
