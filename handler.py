"""PaddleOCR-VL-1.5 RunPod Serverless Handler (vLLM Backend)

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
    3. Detects collapsed table rows -> retries with UVDoc unwarping
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

import asyncio
import base64
import io
import os
import re
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import requests
import runpod
from PIL import Image

# ==========================================================================
# PERFORMANCE OPTIMIZATIONS - Set before any PaddlePaddle imports
# ==========================================================================

os.environ["PADDLEX_SKIP_MODEL_CHECK"] = "1"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", message=".*Non compatible API.*")
warnings.filterwarnings("ignore", category=Warning, module="paddle.utils.decorator_utils")

# Global pipeline - loaded once at container startup
paddle_vl_pipeline = None
doc_preprocessor = None  # DocPreprocessor for server-side preprocessing

def _env_int(name: str, default: int) -> int:
    """Read integer env var with fallback."""
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# Config (baked defaults are set in start.sh/Dockerfile; these allow override)
WORKER_MAX_CONCURRENCY = max(1, _env_int("PADDLE_VL_WORKER_CONCURRENCY", 1))
SERIALIZE_PREDICT = os.environ.get("PADDLE_VL_SERIALIZE", "false").lower() == "true"
MAX_PAGES_PER_BATCH = max(1, _env_int("PADDLE_VL_MAX_PAGES_PER_BATCH", 9))
DOWNLOAD_WORKERS = max(1, _env_int("PADDLE_VL_DOWNLOAD_WORKERS", 20))
USE_QUEUES = os.environ.get("PADDLE_VL_USE_QUEUES", "true").lower() == "true"
VL_REC_MAX_CONCURRENCY = max(1, _env_int("PADDLE_VL_VL_REC_MAX_CONCURRENCY", 20))
# CV_DEVICE=gpu uses paddlepaddle-gpu from base image (paddlex-genai-vllm-server)
# for layout detection (PP-DocLayoutV3). Both VLM and layout run on GPU now.
CV_DEVICE = os.environ.get("CV_DEVICE", os.environ.get("PADDLE_VL_DEVICE", "gpu"))

# Thread pool for parallel image downloads (match concurrency_modifier for job-level parallelism)
_download_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)

# Optional serialization lock for pipeline.predict() across concurrent jobs
_predict_lock = None


def load_pipeline():
    """Load PaddleOCR-VL-1.5 pipeline with vLLM backend + UVDoc pre-loaded."""
    global paddle_vl_pipeline, doc_preprocessor
    if paddle_vl_pipeline is not None:
        return paddle_vl_pipeline

    print(
        "[PaddleOCR-VL] Loading v1.5 pipeline with vLLM backend "
        f"(use_queues={USE_QUEUES}, vl_rec_max_concurrency={VL_REC_MAX_CONCURRENCY}, "
        f"max_pages_per_batch={MAX_PAGES_PER_BATCH}, worker_concurrency={WORKER_MAX_CONCURRENCY}, "
        f"device={CV_DEVICE})..."
    )
    start = time.time()

    from paddleocr import PaddleOCRVL

    # PP-DocLayoutV3 + PaddleOCR-VL-1.5-0.9B (via genai_server with vLLM backend)
    # Per: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
    # device="cpu" forces PP-DocLayoutV3 to run on CPU, avoiding CUDA kernel crashes
    # (PaddlePaddle's pre-compiled CUDA binaries lack kernels for Ampere compute 8.6)

    base_kwargs = {
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": "http://localhost:8080/v1",
        "use_doc_orientation_classify": True,  # Pre-load PP-LCNet orientation model
        "use_doc_unwarping": True,  # Pre-load UVDoc model so retry is fast
    }

    # Not all PaddleOCR versions support all kwargs; try progressively for compatibility.
    candidate_kwargs = [
        {
            **base_kwargs,
            "use_queues": USE_QUEUES,
            "vl_rec_max_concurrency": VL_REC_MAX_CONCURRENCY,
            "device": CV_DEVICE,
        },
        {**base_kwargs, "use_queues": USE_QUEUES, "vl_rec_max_concurrency": VL_REC_MAX_CONCURRENCY},
        {**base_kwargs, "use_queues": USE_QUEUES, "device": CV_DEVICE},
        {**base_kwargs, "use_queues": USE_QUEUES},
        {**base_kwargs, "device": CV_DEVICE},
        base_kwargs,
    ]

    last_err = None
    for kwargs in candidate_kwargs:
        try:
            paddle_vl_pipeline = PaddleOCRVL(**kwargs)
            print(f"[PaddleOCR-VL] PaddleOCRVL init kwargs: {sorted(list(kwargs.keys()))}")
            break
        except TypeError as e:
            last_err = e

    if paddle_vl_pipeline is None:
        raise last_err

    elapsed = time.time() - start
    print(f"[PaddleOCR-VL] Pipeline loaded in {elapsed:.2f}s (vLLM v1.5 backend)")

    # Load DocPreprocessor for server-side preprocessing (orientation + UVDoc)
    # NOTE: Use paddleocr.DocPreprocessor (NOT paddlex.create_pipeline) because it accepts constructor params
    try:
        from paddleocr import DocPreprocessor
        doc_preprocessor = DocPreprocessor(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            device=CV_DEVICE,
        )
        print(f"[PaddleOCR-VL] DocPreprocessor loaded for server-side preprocessing (device={CV_DEVICE})")
    except Exception as e:
        print(f"[PaddleOCR-VL] DocPreprocessor load failed (non-fatal, will skip preprocessing): {e}")
        doc_preprocessor = None

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


def _detect_image_suffix(image_bytes: bytes) -> str:
    # Keep the suffix aligned with content to help downstream decoders.
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return ".img"


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
    """Convert numpy arrays and other non-serializable types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    return obj


def download_image(url: str) -> bytes:
    """Download image from URL and return raw bytes."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def prepare_temp_file(image_bytes: bytes, index: int, skip_resize: bool) -> str:
    """Save image bytes to a temp file, optionally resizing. Returns temp file path."""
    if skip_resize:
        # Fast-path: assume client already produced an OCR-optimized image (e.g. Ghostscript PNG).
        suffix = _detect_image_suffix(image_bytes)
        tmp = tempfile.NamedTemporaryFile(suffix=f"_page{index}{suffix}", delete=False, dir="/tmp")
        with open(tmp.name, "wb") as f:
            f.write(image_bytes)
        tmp.close()
        return tmp.name

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = resize_image_if_needed(image, max_dimension=1920)

    tmp = tempfile.NamedTemporaryFile(suffix=f"_page{index}.png", delete=False, dir="/tmp")
    image.save(tmp.name, "PNG")
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
                md_texts = md_info.get("markdown_texts", "")
                if isinstance(md_texts, str):
                    markdown_output += md_texts
                elif isinstance(md_texts, list):
                    markdown_output += "\n\n".join(str(t) for t in md_texts)
            elif isinstance(md_info, str):
                markdown_output += md_info
    except Exception as e:
        print(f"[PaddleOCR-VL] Error accessing markdown (page {page_number}): {e}")

    try:
        json_data = res.json
        if json_data:
            json_output = convert_to_serializable(json_data)
            if isinstance(json_data, dict) and "parsing_res_list" in json_data:
                parsing_res_list = convert_to_serializable(json_data["parsing_res_list"])
    except Exception as e:
        print(f"[PaddleOCR-VL] Error accessing json (page {page_number}): {e}")

    # Fallback: build markdown from parsing_res_list
    if not markdown_output and parsing_res_list:
        for block in parsing_res_list:
            label = block.get("block_label", "")
            content = block.get("block_content", "")
            if content:
                if label == "table":
                    markdown_output += f"\n\n{content}\n\n"
                else:
                    markdown_output += f"\n{content}\n"

    return {
        "page_number": page_number,
        "markdown": markdown_output.strip(),
        "parsing_res_list": parsing_res_list,
        "json": json_output,
    }


def is_collapsed_page(markdown: str) -> bool:
    """Detect if a page has collapsed table rows.

    Collapsed rows occur when PaddleOCR-VL dumps all transactions into a single
    <tr>, with multiple amounts/dates crammed into individual <td> cells.
    Retrying with doc unwarping fixes this.
    """
    cells = re.findall(r"<td[^>]*>(.*?)</td>", markdown, re.DOTALL)
    for cell in cells:
        text = cell.strip()
        # Multiple dollar amounts in one cell = collapsed
        amounts = re.findall(r"[\d,]+\.\d{2}", text)
        if len(amounts) >= 3:
            return True
        # Multiple MMMDD dates in one cell = collapsed (e.g. DEC08 DEC09 DEC10)
        dates = re.findall(r"[A-Z]{3}\d{2}", text)
        if len(dates) >= 3:
            return True
    return False


def process_batch(
    pipeline,
    batch_paths: list[str],
    *,
    use_orientation: bool = True,
    use_unwarping: bool = False,
) -> list:
    """Process a batch of pages, returns list of results."""
    return list(
        pipeline.predict(
            batch_paths,
            use_doc_orientation_classify=use_orientation,
            use_doc_unwarping=use_unwarping,
        )
    )


async def process_batch_async(
    pipeline,
    batch_paths: list[str],
    *,
    use_orientation: bool = True,
    use_unwarping: bool = False,
) -> list:
    """Async wrapper around batch predict with optional serialization lock."""
    global _predict_lock

    loop = asyncio.get_running_loop()
    fn = partial(
        process_batch,
        pipeline,
        batch_paths,
        use_orientation=use_orientation,
        use_unwarping=use_unwarping,
    )

    if SERIALIZE_PREDICT:
        if _predict_lock is None:
            _predict_lock = asyncio.Lock()
        async with _predict_lock:
            return await loop.run_in_executor(None, fn)

    return await loop.run_in_executor(None, fn)

async def process_pages_with_fallback(pipeline, temp_paths: list[str]) -> list:
    """Process pages with robust fallback strategy.

    Strategy:
    1. Try batch processing (max MAX_PAGES_PER_BATCH pages per batch)
    2. If batch fails, retry once
    3. If still fails, fallback to page-by-page processing to isolate bad inputs

    Returns list of page results in order (None for pages that failed).
    """
    total_pages = len(temp_paths)
    all_results = [None] * total_pages  # Pre-allocate to maintain order

    batches = []
    for i in range(0, total_pages, MAX_PAGES_PER_BATCH):
        batch_end = min(i + MAX_PAGES_PER_BATCH, total_pages)
        batches.append((i, batch_end, temp_paths[i:batch_end]))

    if len(batches) > 1:
        print(
            f"[PaddleOCR-VL] Split {total_pages} pages into {len(batches)} batches "
            f"(max {MAX_PAGES_PER_BATCH}/batch)"
        )

    for batch_idx, (start_idx, end_idx, batch_paths) in enumerate(batches):
        batch_size = len(batch_paths)
        batch_label = f"batch {batch_idx + 1}/{len(batches)}" if len(batches) > 1 else "batch"

        # Try batch processing
        try:
            predict_start = time.time()
            print(f"[PaddleOCR-VL] Processing {batch_label}: {batch_size} page(s)")
            results = await process_batch_async(pipeline, batch_paths, use_orientation=True, use_unwarping=False)
            predict_time = time.time() - predict_start
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} completed in {predict_time:.2f}s")

            for i, res in enumerate(results):
                all_results[start_idx + i] = res
            continue

        except Exception as e:
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} failed: {e}")

        # Retry batch once
        try:
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} failed, retrying entire batch")
            await asyncio.sleep(1)
            retry_start = time.time()
            results = await process_batch_async(pipeline, batch_paths, use_orientation=True, use_unwarping=False)
            retry_time = time.time() - retry_start
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} retry completed in {retry_time:.2f}s")

            for i, res in enumerate(results):
                all_results[start_idx + i] = res
            continue

        except Exception as e:
            print(f"[PaddleOCR-VL] {batch_label.capitalize()} retry also failed: {e}")

        # Fallback to page-by-page
        print(f"[PaddleOCR-VL] Falling back to page-by-page for {batch_label} ({batch_size} page(s))")
        for i, path in enumerate(batch_paths):
            page_num = start_idx + i + 1
            try:
                single_results = await process_batch_async(pipeline, [path], use_orientation=True, use_unwarping=False)
                all_results[start_idx + i] = single_results[0] if single_results else None
            except Exception as e:
                print(f"[PaddleOCR-VL] Page {page_num} failed: {e}")

    return all_results


async def handler(event):
    """RunPod serverless handler (async, concurrent-capable)."""
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
                    "concurrent": True,
                    "serialize_predict": SERIALIZE_PREDICT,
                    "max_pages_per_batch": MAX_PAGES_PER_BATCH,
                    "download_workers": DOWNLOAD_WORKERS,
                    "worker_max_concurrency": WORKER_MAX_CONCURRENCY,
                    "use_queues": USE_QUEUES,
                    "vl_rec_max_concurrency": VL_REC_MAX_CONCURRENCY,
                    "cv_device": CV_DEVICE,
                    "doc_preprocessor_available": doc_preprocessor is not None,
                },
            }

        skip_resize = job_input.get("skip_resize", False)
        # NOTE: Preprocessing (orientation + UVDoc) is now done on the backend server
        # RunPod receives already-preprocessed images and only runs VLM inference

        # Collect image bytes from URLs or base64
        image_bytes_list: list[bytes] = []

        # Priority 1: image_urls (preferred - avoids base64 overhead)
        image_urls = job_input.get("image_urls", [])
        if image_urls:
            print(
                f"[PaddleOCR-VL] Downloading {len(image_urls)} images from URLs "
                f"(parallel, workers={DOWNLOAD_WORKERS})..."
            )
            dl_start = time.time()
            loop = asyncio.get_running_loop()
            futures = [loop.run_in_executor(_download_pool, download_image, url) for url in image_urls]
            image_bytes_list = list(await asyncio.gather(*futures))
            print(f"[PaddleOCR-VL] Downloaded {len(image_bytes_list)} image(s) in {time.time() - dl_start:.2f}s")

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
                "error": "No images provided. Send 'image_urls', 'images_base64', or 'image_base64'.",
            }

        if skip_resize:
            print("[PaddleOCR-VL] skip_resize=True (client handled sizing)")

        print(f"[PaddleOCR-VL] Processing {len(image_bytes_list)} page(s) (vLLM v1.5 backend)")

        pipeline = load_pipeline()

        temp_paths: list[str] = []
        original_temp_paths: list[str] = []  # Track original paths for cleanup if preprocessing
        try:
            prep_start = time.time()
            for i, img_bytes in enumerate(image_bytes_list):
                temp_paths.append(prepare_temp_file(img_bytes, i + 1, skip_resize))
            print(
                f"[PaddleOCR-VL] Prepared {len(temp_paths)} temp file(s) in {time.time() - prep_start:.2f}s "
                f"(skip_resize={skip_resize})"
            )

            # NOTE: Preprocessing (orientation + UVDoc) is now done on the backend server
            # Images arriving here are already preprocessed, so we skip to VLM inference

            # Pass 1: Batch predict with orientation auto-detection, WITHOUT doc unwarping
            predict_start = time.time()
            results = await process_pages_with_fallback(pipeline, temp_paths)
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
                        "json": None,
                    }
                    print(f"[PaddleOCR-VL] Page {i + 1} FAILED - empty result")
                else:
                    page_result = extract_page_result(res, page_number=i + 1)
                    print(f"[PaddleOCR-VL] Page {i + 1} markdown length: {len(page_result['markdown'])}")
                    if is_collapsed_page(page_result["markdown"]):
                        collapsed_indices.append(i)

                pages.append(page_result)

            # Pass 2: Retry collapsed pages WITH doc unwarping
            # UVDoc can crash with "cv worker: std::exception" on certain images
            # See: https://github.com/PaddlePaddle/PaddleOCR/issues/17206
            if collapsed_indices:
                print(
                    f"[PaddleOCR-VL] {len(collapsed_indices)} collapsed page(s) detected: "
                    f"{[i + 1 for i in collapsed_indices]}, retrying with doc unwarping"
                )
                retry_paths = [temp_paths[i] for i in collapsed_indices]

                retry_success = False
                for attempt in range(1, 3):
                    try:
                        retry_start = time.time()
                        retry_results = await process_batch_async(
                            pipeline,
                            retry_paths,
                            use_orientation=True,
                            use_unwarping=True,
                        )
                        retry_time = time.time() - retry_start
                        print(
                            f"[PaddleOCR-VL] Doc unwarping retry completed in {retry_time:.2f}s for "
                            f"{len(retry_paths)} page(s) (attempt {attempt})"
                        )

                        for j, orig_idx in enumerate(collapsed_indices):
                            page_result = extract_page_result(retry_results[j], page_number=orig_idx + 1)
                            print(
                                f"[PaddleOCR-VL] Page {orig_idx + 1} retried: markdown length "
                                f"{len(page_result['markdown'])}"
                            )
                            pages[orig_idx] = page_result

                        retry_success = True
                        break

                    except Exception as e:
                        print(f"[PaddleOCR-VL] Doc unwarping attempt {attempt} failed: {e}")
                        if attempt < 2:
                            print("[PaddleOCR-VL] Retrying doc unwarping...")
                            await asyncio.sleep(1)

                if not retry_success:
                    print(
                        "[PaddleOCR-VL] Doc unwarping failed after 2 attempts, "
                        "keeping original results for collapsed pages"
                    )

        finally:
            # Cleanup all temp files (processed + original if preprocessing was used)
            all_cleanup_paths = set(temp_paths) | set(original_temp_paths)
            for tmp_path in all_cleanup_paths:
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
                "processingTime": processing_time,
            },
        }

    except Exception as e:
        import traceback

        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"[PaddleOCR-VL] Error: {error_msg}\n{stack_trace}")
        return {
            "status": "error",
            "error": error_msg,
            "stack_trace": stack_trace,
        }


def concurrency_modifier(current_concurrency: int) -> int:
    """Worker-level job concurrency.

    Keep this low for stability; page/VLM concurrency is controlled separately by
    MAX_PAGES_PER_BATCH and VL_REC_MAX_CONCURRENCY.
    """
    return WORKER_MAX_CONCURRENCY


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
    }
)
