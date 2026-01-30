"""
One-off warmup helper to pre-download PaddleOCR-VL models into the RunPod volume.

Run this on a GPU pod with the network volume attached, then shut it down.
Serverless containers will reuse the cached models for faster cold starts.
"""

import os
import time


def setup_model_cache() -> bool:
    """Configure model cache directory for faster cold starts."""
    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    model_cache_dir = os.environ.get(
        "PADDLE_VL_CACHE_DIR",
        os.path.join(network_volume_path, "paddle_models"),
    )

    if os.path.exists(network_volume_path) and os.access(network_volume_path, os.W_OK):
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["PADDLE_HOME"] = model_cache_dir
        os.environ["PADDLEOCR_HOME"] = model_cache_dir
        os.environ["HF_HOME"] = os.path.join(model_cache_dir, "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(model_cache_dir, "huggingface", "hub")
        print(f"[Warmup] Using network volume cache: {model_cache_dir}")
        return True

    print("[Warmup] No network volume found, using container storage")
    return False


def main() -> None:
    setup_model_cache()

    # Also set vLLM cache to network volume
    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if os.path.exists(network_volume_path):
        vllm_cache = os.path.join(network_volume_path, "vllm_cache")
        os.makedirs(vllm_cache, exist_ok=True)
        os.environ.setdefault("VLLM_CACHE_ROOT", vllm_cache)
        print(f"[Warmup] vLLM cache dir: {vllm_cache}")

    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

    start = time.time()
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL()
    elapsed = time.time() - start
    print(f"[Warmup] Models ready in {elapsed:.2f}s")

    # Run a dummy inference to populate torch.compile cache
    import tempfile
    from PIL import Image

    try:
        warmup_start = time.time()
        dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_img.save(tmp.name, "PNG")
            dummy_path = tmp.name
        for _ in pipeline.predict(dummy_path):
            pass
        os.unlink(dummy_path)
        print(f"[Warmup] Dummy inference done in {time.time() - warmup_start:.2f}s")
        print(f"[Warmup] torch.compile cache should now be populated on network volume")
    except Exception as e:
        print(f"[Warmup] Dummy inference failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
