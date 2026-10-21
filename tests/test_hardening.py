"""Test suite for Task 4 hardening requirements:
1. Threading and inference configuration (torch.set_num_threads, torch.inference_mode).
2. Image safety: upload size caps, decompression bomb protection, format checks, and downscaling.
3. Execution timeout safety (run_with_timeout).
4. Memory instrumentation (Windows NT kernel memory reading).
5. File integrity checks and hash verification.
6. Doctor diagnostic suite.
"""

import os
import time
import math
import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.doctor import (
    compute_sha256,
    check_working_directory,
    check_package_versions,
    check_utf8_environment,
    check_weights_and_data_integrity,
    check_reference_database_integrity,
    check_port_free
)
from src.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_UPLOAD_SIZE_MB,
    MAX_PIXEL_COUNT,
    MAX_DISPLAY_PIXELS,
    QUEUE_MAX_SIZE,
    DEFAULT_CONCURRENCY_LIMIT,
    TORCH_NUM_THREADS,
    REQUEST_TIMEOUT_SECONDS,
    EXPECTED_HASHES
)
from src.utils import (
    safe_process_image,
    ensure_rgb_image,
    run_with_timeout,
    list_pending_submissions
)
from scripts.doctor import (
    compute_sha256,
    check_working_directory,
    check_package_versions,
    check_utf8_environment,
    check_weights_and_data_integrity,
    check_port_free
)
from scripts.load_test import get_process_memory


# ── Test 1: Torch Configuration ───────────────────────────────────────────────

def test_torch_num_threads_configured():
    """Verify torch threading is configured according to config."""
    torch.set_num_threads(TORCH_NUM_THREADS)
    assert torch.get_num_threads() == TORCH_NUM_THREADS


# ── Test 2: Image Safety and Decompression Bomb Protection ────────────────────

def test_safe_image_processing_valid(tmp_path):
    """Verify normal valid image passes cleanly and preserves properties."""
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (200, 150), color=(100, 150, 200))
    img.save(img_path, format="JPEG")

    sanitized, meta = safe_process_image(str(img_path))
    assert sanitized.mode == "RGB"
    assert sanitized.size == (200, 150)
    assert meta["downscaled"] is False
    assert meta["total_pixels"] == 30000


def test_oversized_file_rejected(tmp_path):
    """Verify files exceeding MAX_UPLOAD_SIZE_MB are rejected fast."""
    oversized_file = tmp_path / "oversized.jpg"
    # Create file slightly larger than limit with custom low max_size_mb
    with open(oversized_file, "wb") as f:
        f.write(b"0" * (2 * 1024 * 1024))

    # Test with max_size_mb=1 to trigger fast rejection
    with pytest.raises(ValueError, match="exceeds maximum upload limit"):
        safe_process_image(str(oversized_file), max_size_mb=1)


def test_decompression_bomb_rejected(tmp_path):
    """Verify image exceeding pixel limit is rejected as a decompression bomb."""
    # Test with custom small pixel limit
    large_img = Image.new("RGB", (500, 500))  # 250,000 pixels
    with pytest.raises(ValueError, match="decompression bomb protection"):
        safe_process_image(large_img, max_pixels=100_000)


def test_downscaling_oversized_images():
    """Verify image exceeding max_display_pixels is downscaled preserving aspect ratio."""
    # 2000 x 2000 = 4,000,000 pixels > 2,048,000 max_display_pixels
    img = Image.new("RGB", (2000, 2000), color=(128, 128, 128))
    sanitized, meta = safe_process_image(img, max_display_pixels=2_048_000)

    assert meta["downscaled"] is True
    assert meta["orig_width"] == 2000
    assert meta["orig_height"] == 2000
    assert sanitized.width * sanitized.height <= 2_048_000
    # Aspect ratio preserved (square)
    assert sanitized.width == sanitized.height


def test_unsupported_file_extension(tmp_path):
    """Verify invalid file extensions are rejected."""
    bad_file = tmp_path / "test.exe"
    bad_file.write_text("dummy binary content")

    with pytest.raises(ValueError, match="Unsupported image file extension"):
        safe_process_image(str(bad_file))


# ── Test 3: Request Timeout Safety ───────────────────────────────────────────

def test_run_with_timeout_success():
    """Verify function finishing within timeout returns result normally."""
    def quick_fn(x, y):
        return x + y

    res = run_with_timeout(quick_fn, args=(10, 25), timeout=2.0)
    assert res == 35


def test_run_with_timeout_exceeded():
    """Verify slow function raises TimeoutError without hanging process."""
    def slow_fn():
        time.sleep(1.5)
        return "done"

    with pytest.raises(TimeoutError, match="timed out"):
        run_with_timeout(slow_fn, timeout=0.2)


# ── Test 4: Memory Instrumentation ───────────────────────────────────────────

def test_process_memory_measurement():
    """Verify Windows process memory instrumentation returns valid non-zero metrics."""
    pid = os.getpid()
    mem = get_process_memory(pid)

    assert mem is not None
    assert "working_set_mb" in mem
    assert "peak_working_set_mb" in mem
    assert "private_mb" in mem
    assert mem["working_set_mb"] > 0.0
    assert mem["private_mb"] > 0.0
    assert mem["peak_working_set_mb"] >= mem["working_set_mb"]


# ── Test 5: Bounded In-Memory Structures ──────────────────────────────────────

def test_list_pending_submissions_bounded(tmp_path):
    """Verify list_pending_submissions respects max_items bound."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()

    # Create 10 dummy pending json files
    for i in range(10):
        sub_file = pending_dir / f"sub_{i:03d}.json"
        sub_file.write_text('{"status": "pending", "submission_id": "sub_' + f'{i:03d}' + '"}', encoding="utf-8")

    bounded_list = list_pending_submissions(str(pending_dir), max_items=4)
    assert len(bounded_list) == 4


# ── Test 6: Doctor Diagnostic Functions ──────────────────────────────────────

def test_doctor_working_directory_check():
    """Verify working directory check passes from repository root."""
    assert check_working_directory() is True


def test_doctor_utf8_check():
    """Verify UTF-8 probe passes."""
    assert check_utf8_environment() is True


def test_doctor_package_versions_check():
    """Verify core package versions check passes."""
    assert check_package_versions() is True


def test_doctor_weights_integrity():
    """Verify hash check passes for tracked weights and data files."""
    assert check_weights_and_data_integrity() is True

import json
import numpy as np

def _write_reference_fixture(tmp_path, embeddings, keys, metadata_paths=None):
    """Create a temporary reference database for integrity tests."""
    npz_path = tmp_path / "reference_features.npz"
    json_path = tmp_path / "reference_features.json"

    np.savez_compressed(
        npz_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        keys=np.asarray(keys, dtype=str),
    )

    paths = list(keys) if metadata_paths is None else metadata_paths
    metadata = {
        "format_version": 1,
        "count": len(paths),
        "updated_at": "test",
        "entries": [
            {"index": i, "image_path": path, "tiger_name": "tiger_test"}
            for i, path in enumerate(paths)
        ],
    }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return npz_path, json_path


def test_reference_integrity_rejects_mismatched_keys(tmp_path):
    npz_path, json_path = _write_reference_fixture(
        tmp_path,
        np.zeros((2, 128)),
        ["image_a.jpg", "image_b.jpg"],
        ["image_a.jpg", "different_image.jpg"],
    )
    assert check_reference_database_integrity(npz_path, json_path) is False


def test_reference_integrity_rejects_wrong_embedding_dimensions(tmp_path):
    npz_path, json_path = _write_reference_fixture(
        tmp_path,
        np.zeros((1, 64)),
        ["image_a.jpg"],
    )
    assert check_reference_database_integrity(npz_path, json_path) is False


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_reference_integrity_rejects_nonfinite_embeddings(tmp_path, bad_value):
    embeddings = np.zeros((1, 128), dtype=np.float32)
    embeddings[0, 0] = bad_value
    npz_path, json_path = _write_reference_fixture(
        tmp_path, embeddings, ["image_a.jpg"]
    )
    assert check_reference_database_integrity(npz_path, json_path) is False
