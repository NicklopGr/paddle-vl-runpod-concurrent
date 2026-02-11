"""
PaddleOCR-VL-1.5 RunPod Serverless Handler (vLLM Backend)

Pipeline:
1. PP-LCNet (Orientation) - Auto-detects and corrects 0°/90°/180°/270° rotation
2. PP-DocLayoutV3 (Layout Analysis) - Detects 25 element categories with reading order
3. PaddleOCR-VL-1.5-0.9B via vLLM server (continuous batching, Flash Attention)
4. Post-processing - Outputs structured markdown with HTML tables
5. UVDoc (Fallback) - Re-processes pages with collapsed table rows using doc unwarping

Architecture:
  - vLLM serves PaddleOCR-VL-1.5-0.9B on localhost:8080 (started by start.sh)
  - This handler uses PaddleOCRVL pipeline client which:
    1. Runs PP-DocLayoutV3 layout detection (local, CPU)
    2. Sends cropped regions to vLLM server (batched VLM inference)
    3. Detects collapsed table rows → retries with UVDoc unwarping
    4. Assembles markdown output

Input:
{
    "input": {
        "image_urls": ["https://...", ...],   // URL mode (preferred)
        "images_base64": ["base64_1", ...],   // OR base64 mode
        "image_base64": "base64_single",      // OR single image
        "skip_resize": false,
        "warmup": true
    }
}

Output:
{
    "status": "success",
    "result": {
        "pages": [{ "page_number": 1, "markdown": "...", "parsing_res_list": [...], "json": {...} }],
        "ocrProvider": "paddleocr-vl-vllm",
        "processingTime": 1234
    }
}
"""

import runpod
import asyncio
import base64
import tempfile
import os
import time
import re
import warnings
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import numpy as np

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Set before any PaddlePaddle imports
# ============================================================================

os.environ["PADDLEX_SKIP_MODEL_CHECK"] = "1"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", message=".*Non compatible API.*")
warnings.filterwarnings("ignore", category=Warning, module="paddle.utils.decorator_utils")

# Global pipeline - loaded once at container startup
paddle_vl_pipeline = None

# Thread pool for parallel image downloads (match MAX_PAGES_PER_BATCH)
_download_pool = ThreadPoolExecutor(max_workers=10)


def load_pipeline():
    """Load PaddleOCR-VL-1.5 pipeline with vLLM backend + UVDoc pre-loaded"""
    global paddle_vl_pipeline
    if paddle_vl_pipeline is not None:
        return paddle_vl_pipeline

    print("[PaddleOCR-VL] Loading v1.5 pipeline with vLLM backend...")
    start = time.time()

    from paddleocr import PaddleOCRVL

    # PP-DocLayoutV3 + PaddleOCR-VL-1.5-0.9B (via genai_server with vLLM backend)
    # Per: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
    # device="cpu" forces PP-DocLayoutV3 to run on CPU, avoiding CUDA kernel crashes
    # (PaddlePaddle's pre-compiled CUDA binaries lack kernels for Ampere compute 8.6)
    paddle_vl_pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://localhost:8080/v1",
        use_doc_orientation_classify=True,  # Pre-load PP-LCNet orientation model
        use_doc_unwarping=True,  # Pre-load UVDoc model so retry is fast
        use_queues=True,  # Enable queue-based concurrent execution (thread-safe)
        vl_rec_max_concurrency=10,  # Match MAX_PAGES_PER_BATCH (reduced from 20 to test cv worker stability)
        device="cpu",  # Force CPU for layout detection (avoids cv worker crashes)
    )

    elapsed = time.time() - start
    print(f"[PaddleOCR-VL] Pipeline loaded in {elapsed:.2f}s (vLLM v1.5 backend)")

    # Run a dummy inference to warm up vLLM (avoids ~35s penalty on first real job)
    try:
        warmup_start = time.time()
        dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_img.save(tmp.name, "PNG")
            dummy_path = tmp.name
        for _ in paddle_vl_pipeline.predict(dummy_path, use_doc_unwarping=False):
            pass
        os.unlink(dummy_path)
        print(f"[PaddleOCR-VL] Warmup inference done in {time.time() - warmup_start:.2f}s")
    except Exception as e:
        print(f"[PaddleOCR-VL] Warmup inference failed (non-fatal): {e}")

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
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def prepare_temp_file(image_bytes: bytes, index: int, skip_resize: bool) -> str:
    """Save image bytes to a temp file, optionally resizing. Returns temp file path."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if not skip_resize:
        image = resize_image_if_needed(image, max_dimension=1920)

    tmp = tempfile.NamedTemporaryFile(suffix=f'_page{index}.png', delete=False, dir='/tmp')
    image.save(tmp.name, 'PNG')
    tmp.close()
    return tmp.name


def extract_page_result(res, page_number: int) -> dict:
    """Extract markdown and structured data from a single PaddleOCR-VL result."""
    markdown_output = ""
    parsing_res_list = []
    json_output = None

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
        print(f"[PaddleOCR-VL] Error accessing markdown (page {page_number}): {e}")

    try:
        json_data = res.json
        if json_data:
            json_output = convert_to_serializable(json_data)
            if isinstance(json_data, dict) and 'parsing_res_list' in json_data:
                parsing_res_list = convert_to_serializable(json_data['parsing_res_list'])
    except Exception as e:
        print(f"[PaddleOCR-VL] Error accessing json (page {page_number}): {e}")

    # Fallback: build markdown from parsing_res_list
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


def is_collapsed_page(markdown: str) -> bool:
    """Detect if a page has collapsed table rows.

    Collapsed rows occur when PaddleOCR-VL dumps all transactions into a single
    <tr>, with multiple amounts/dates crammed into individual <td> cells.
    Retrying with doc unwarping fixes this.
    """
    cells = re.findall(r'<td[^>]*>(.*?)</td>', markdown, re.DOTALL)
    for cell in cells:
        text = cell.strip()
        # Multiple dollar amounts in one cell = collapsed
        amounts = re.findall(r'[\d,]+\.\d{2}', text)
        if len(amounts) >= 3:
            return True
        # Multiple MMMDD dates in one cell = collapsed (e.g. DEC08 DEC09 DEC10)
        dates = re.findall(r'[A-Z]{3}\d{2}', text)
        if len(dates) >= 3:
            return True
    return False


# ============================================================================
# BATCH PROCESSING WITH RETRY AND FALLBACK
# ============================================================================

# Max pages per batch - reduced from 20 to 10 to investigate cv worker crashes
# cv worker crashes may be caused by memory pressure at higher concurrency
MAX_PAGES_PER_BATCH = 10


def process_batch(pipeline, batch_paths: list[str], use_orientation: bool = True, use_unwarping: bool = False) -> list:
    """Process a batch of pages, returns list of results."""
    return list(pipeline.predict(
        batch_paths,
        use_doc_orientation_classify=use_orientation,
        use_doc_unwarping=use_unwarping
    ))


def process_pages_with_fallback(pipeline, temp_paths: list[str]) -> list:
    """
    Process pages with robust fallback strategy:
    1. Try batch processing (max 10 pages per batch)
    2. If batch fails, retry once
    3. If still fails, fallback to page-by-page processing

    Returns list of page results in order.
    """
    total_pages = len(temp_paths)
    all_results = [None] * total_pages  # Pre-allocate to maintain order

    # Split into batches of MAX_PAGES_PER_BATCH
    batches = []
    for i in range(0, total_pages, MAX_PAGES_PER_BATCH):
        batch_end = min(i + MAX_PAGES_PER_BATCH, total_pages)
        batches.append((i, batch_end, temp_paths[i:batch_end]))

    if len(batches) > 1:
        print(f"[PaddleOCR-VL] Split {total_pages} pages into {len(batches)} batches (max {MAX_PAGES_PER_BATCH}/batch)")

    for batch_idx, (start_idx, end_idx, batch_paths) in enumerate(batches):
        batch_size = len(batch_paths)
        batch_label = f"batch {batch_idx + 1}/{len(batches)}" if len(batches) > 1 else "batch"

        # Try batch processing (no retry for cv worker errors - go straight to page-by-page)
        batch_success = False
        try:
            predict_start = time.time()
            print(f"[PaddleOCR-VL] Processing {batch_label}: {batch_size} page(s)")

            results = process_batch(pipeline, batch_paths, use_orientation=True, use_unwarping=False)

            predict_time = time.time() - predict_start
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} completed in {predict_time:.2f}s")

            # Store results in correct positions
            for i, res in enumerate(results):
                all_results[start_idx + i] = res

            batch_success = True

        except Exception as e:
            error_msg = str(e)
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} failed: {error_msg}")
            # cv worker errors are deterministic - retrying same batch won't help
            # Go straight to page-by-page to isolate the problematic image

        # If batch failed, retry the whole batch once (cv worker crashes are fixed with device="cpu")
        if not batch_success:
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} failed, retrying entire batch")

            try:
                time.sleep(1)  # Brief pause before retry
                retry_start = time.time()
                results = process_batch(pipeline, batch_paths, use_orientation=True, use_unwarping=False)

                retry_time = time.time() - retry_start
                print(f"[PaddleOCR-VL] {batch_label.capitalize()} retry completed in {retry_time:.2f}s")

                for i, res in enumerate(results):
                    all_results[start_idx + i] = res
                batch_success = True

            except Exception as e:
                print(f"[PaddleOCR-VL] {batch_label.capitalize()} retry also failed: {e}")
                # Leave results as None - will be handled as empty pages

    return all_results


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
                    "model": "PaddleOCR-VL-1.5",
                    "concurrent": True
                }
            }

        skip_resize = job_input.get("skip_resize", False)

        # Collect image bytes from URLs or base64
        image_bytes_list: list[bytes] = []

        # Priority 1: image_urls (preferred - avoids base64 overhead)
        image_urls = job_input.get("image_urls", [])
        if image_urls:
            print(f"[PaddleOCR-VL] Downloading {len(image_urls)} images from URLs (parallel)...")
            dl_start = time.time()
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(_download_pool, download_image, url) for url in image_urls]
            image_bytes_list = list(await asyncio.gather(*futures))
            print(f"[PaddleOCR-VL] All {len(image_urls)} images downloaded in {time.time() - dl_start:.2f}s")

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

        print(f"[PaddleOCR-VL] Processing {len(image_bytes_list)} page(s) (vLLM v1.5 backend)")

        pipeline = load_pipeline()

        # Save all images to temp files for batch predict
        temp_paths = []
        try:
            for i, img_bytes in enumerate(image_bytes_list):
                tmp_path = prepare_temp_file(img_bytes, i + 1, skip_resize)
                temp_paths.append(tmp_path)

            # Pass 1: Batch predict with orientation auto-detection, WITHOUT doc unwarping
            # Uses batching (max 10 pages) with retry and page-by-page fallback
            # PP-LCNet detects 0°/90°/180°/270° rotation and auto-corrects
            predict_start = time.time()
            results = process_pages_with_fallback(pipeline, temp_paths)
            predict_time = time.time() - predict_start
            print(f"[PaddleOCR-VL] All pages processed in {predict_time:.2f}s")

            # Map results to pages and detect collapsed rows
            pages = []
            collapsed_indices = []
            for i, res in enumerate(results):
                if res is None:
                    # Page failed completely - create empty result
                    page_result = {
                        "page_number": i + 1,
                        "markdown": "",
                        "parsing_res_list": [],
                        "json": None
                    }
                    print(f"[PaddleOCR-VL] Page {i+1} FAILED - empty result")
                else:
                    page_result = extract_page_result(res, page_number=i + 1)
                    print(f"[PaddleOCR-VL] Page {i+1} markdown length: {len(page_result['markdown'])}")
                    if is_collapsed_page(page_result['markdown']):
                        collapsed_indices.append(i)
                pages.append(page_result)

            # Pass 2: Retry collapsed pages WITH doc unwarping
            # UVDoc can crash with "cv worker: std::exception" on certain images
            # See: https://github.com/PaddlePaddle/PaddleOCR/issues/17206
            # Strategy: Try up to 2 times, fall back to original if both fail
            if collapsed_indices:
                print(f"[PaddleOCR-VL] {len(collapsed_indices)} collapsed page(s) detected: {[i+1 for i in collapsed_indices]}, retrying with doc unwarping")
                retry_paths = [temp_paths[i] for i in collapsed_indices]

                retry_success = False
                for attempt in range(1, 3):  # Try up to 2 times
                    try:
                        retry_start = time.time()
                        retry_results = list(pipeline.predict(retry_paths, use_doc_unwarping=True))
                        retry_time = time.time() - retry_start
                        print(f"[PaddleOCR-VL] Doc unwarping retry completed in {retry_time:.2f}s for {len(retry_paths)} page(s) (attempt {attempt})")

                        for j, orig_idx in enumerate(collapsed_indices):
                            page_result = extract_page_result(retry_results[j], page_number=orig_idx + 1)
                            print(f"[PaddleOCR-VL] Page {orig_idx+1} retried: markdown length {len(page_result['markdown'])}")
                            pages[orig_idx] = page_result

                        retry_success = True
                        break  # Success, exit retry loop

                    except Exception as e:
                        print(f"[PaddleOCR-VL] Doc unwarping attempt {attempt} failed: {e}")
                        if attempt < 2:
                            print(f"[PaddleOCR-VL] Retrying doc unwarping...")
                            time.sleep(1)  # Brief pause before retry

                if not retry_success:
                    print(f"[PaddleOCR-VL] Doc unwarping failed after 2 attempts, keeping original results for collapsed pages")

        finally:
            # Cleanup all temp files
            for tmp_path in temp_paths:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

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
    Allow up to 10 concurrent jobs.
    Reduced from 20 to investigate cv worker stability issues.
    """
    return 10


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
