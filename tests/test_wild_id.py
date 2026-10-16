"""Pytest test suite for Wild-ID fixes:
1. RGBA and palette (P) image upload handling in memory
2. Strict identity-name validation and traversal prevention
3. Atomic .npz + JSON feature persistence
4. Legacy pickle database migration
5. Pending submission workflow (no matching until reviewed)
6. Safe deletion via trash/
7. Terminology and honesty disclaimers
"""

import os
import json
import time
import shutil
import pickle
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SIMILARITY_THRESHOLD,
    THRESHOLD_LABEL,
    TOP_K_CANDIDATES,
    SIMILARITY_LABEL,
    SIMILARITY_NOTE,
    OUTCOME_ABOVE_THRESHOLD,
    OUTCOME_BELOW_THRESHOLD
)
from src.utils import (
    validate_identity_name,
    ensure_rgb_image,
    save_features_npz,
    load_features_npz,
    submit_to_pending,
    list_pending_submissions,
    review_pending_submission,
    move_to_trash
)
from src.matcher import TigerMatcher
from migrate_database import migrate_pickle_to_npz


# ── Test 1: RGBA and P Mode Upload Handling ───────────────────────────────────

def test_rgba_and_palette_conversion():
    """Ensure RGBA and Palette (P) images convert to RGB cleanly in-memory."""
    # Create RGBA image with alpha channel
    rgba_img = Image.new("RGBA", (100, 100), color=(255, 128, 0, 128))
    rgb_converted = ensure_rgb_image(rgba_img)
    assert rgb_converted.mode == "RGB"
    assert rgb_converted.size == (100, 100)

    # Create P (palette) image
    palette_img = Image.new("P", (100, 100))
    palette_converted = ensure_rgb_image(palette_img)
    assert palette_converted.mode == "RGB"

    # Verify saving converted RGB to JPEG does not raise OSError
    test_jpeg_path = Path("trash") / "test_rgba_save.jpg"
    test_jpeg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rgb_converted.save(test_jpeg_path, format="JPEG")
        assert test_jpeg_path.exists()
    finally:
        if test_jpeg_path.exists():
            test_jpeg_path.unlink()


def test_feature_extractor_rgba_and_weights_only():
    """Verify TigerFeatureExtractor works with weights_only=True and accepts RGBA PIL image in memory."""
    from src.feature_extractor import TigerFeatureExtractor
    extractor = TigerFeatureExtractor(device="cpu")
    rgba_img = Image.new("RGBA", (224, 224), color=(200, 100, 50, 200))
    features = extractor.extract_features(rgba_img)
    assert features is not None
    assert features.shape == (128,)
    assert np.isclose(np.linalg.norm(features), 1.0, atol=1e-3)


def test_backward_compatibility_aliases():
    """Verify legacy SnowLeopard* and identify_leopard aliases are preserved."""
    from src.feature_extractor import SnowLeopardFeatureExtractor, TigerFeatureExtractor
    from src.matcher import SnowLeopardMatcher, TigerMatcher
    import demo

    assert SnowLeopardFeatureExtractor is TigerFeatureExtractor
    assert SnowLeopardMatcher is TigerMatcher
    assert hasattr(demo, "identify_leopard")
    assert demo.identify_leopard is demo.identify_tiger


def test_no_residual_temp_query_on_disk():
    """Verify that root temp_query.jpg is not used or left behind."""
    assert not Path("temp_query.jpg").exists()


# ── Test 2: Strict Name Validation & Traversal ────────────────────────────────

def test_strict_identity_name_validation(tmp_path):
    """Test identity name validation rules and path traversal protection."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()

    # Valid names
    valid_names = ["tiger_001", "Tiger-A", "TIGER_99", "tiger-123_xyz", "t1"]
    for name in valid_names:
        is_valid, msg = validate_identity_name(name, base_dir=str(ref_dir))
        assert is_valid is True, f"Expected '{name}' to be valid, got: {msg}"
        assert msg == name

    # Invalid names: path traversals
    traversal_names = [
        "../tiger",
        "..\\tiger",
        "../../etc/passwd",
        "tiger/nested",
        "tiger\\nested",
        ".",
        "..",
        "/"
    ]
    for name in traversal_names:
        is_valid, msg = validate_identity_name(name, base_dir=str(ref_dir))
        assert is_valid is False, f"Expected traversal '{name}' to be rejected"

    # Invalid characters
    invalid_chars = ["tiger 001", "tiger$01", "tiger*", "tiger?", "tiger@home", "tiger\0null"]
    for name in invalid_chars:
        is_valid, msg = validate_identity_name(name, base_dir=str(ref_dir))
        assert is_valid is False, f"Expected invalid chars in '{name}' to be rejected"

    # Boundary lengths
    is_valid, msg = validate_identity_name("", base_dir=str(ref_dir))
    assert is_valid is False

    is_valid, msg = validate_identity_name("   ", base_dir=str(ref_dir))
    assert is_valid is False

    too_long = "a" * 65
    is_valid, msg = validate_identity_name(too_long, base_dir=str(ref_dir))
    assert is_valid is False

    # Reserved Windows device names
    for reserved in ["CON", "prn", "AUX", "nul", "COM1", "lpt1"]:
        is_valid, msg = validate_identity_name(reserved, base_dir=str(ref_dir))
        assert is_valid is False, f"Expected reserved name '{reserved}' to be rejected"


# ── Test 3: Atomic NPZ and JSON Persistence ───────────────────────────────────

def test_atomic_npz_and_json_save_load(tmp_path):
    """Verify atomic write, clean load with allow_pickle=False, and data fidelity."""
    npz_path = str(tmp_path / "features.npz")
    json_path = str(tmp_path / "features.json")

    # Generate synthetic feature embeddings
    rng = np.random.default_rng(42)
    sample_data = {
        "data/reference/tiger_001/img_001.jpg": rng.normal(size=128).astype(np.float32),
        "data/reference/tiger_001/img_002.jpg": rng.normal(size=128).astype(np.float32),
        "data/reference/tiger_002/img_001.jpg": rng.normal(size=128).astype(np.float32),
    }

    # Normalize vectors
    for k in sample_data:
        sample_data[k] = sample_data[k] / np.linalg.norm(sample_data[k])

    # Atomic save
    save_features_npz(sample_data, npz_path=npz_path, json_path=json_path)

    assert Path(npz_path).exists()
    assert Path(json_path).exists()

    # Verify no temp files left in directory
    leftover_tmps = list(tmp_path.glob("*.tmp*"))
    assert len(leftover_tmps) == 0, f"Found leftover temp files: {leftover_tmps}"

    # Load back with allow_pickle=False
    loaded = load_features_npz(npz_path=npz_path, json_path=json_path)
    assert len(loaded) == len(sample_data)

    for k, v in sample_data.items():
        assert k in loaded
        assert np.allclose(v, loaded[k], atol=1e-6)

    # Verify JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    assert meta["format_version"] == 1
    assert meta["count"] == 3
    assert len(meta["entries"]) == 3
    assert meta["entries"][0]["tiger_name"] == "tiger_001"


# ── Test 4: Database Migration ────────────────────────────────────────────────

def test_database_migration(tmp_path):
    """Verify one-time migration from .pkl to .npz + .json."""
    pkl_path = str(tmp_path / "legacy.pkl")
    npz_path = str(tmp_path / "migrated.npz")
    json_path = str(tmp_path / "migrated.json")

    rng = np.random.default_rng(123)
    original_dict = {
        "data/reference/tiger_100/img_001.jpg": rng.random(128).astype(np.float32),
        "data/reference/tiger_100/img_002.jpg": rng.random(128).astype(np.float32),
        "data/reference/tiger_200/img_001.jpg": rng.random(128).astype(np.float32),
    }

    with open(pkl_path, 'wb') as f:
        pickle.dump(original_dict, f)

    success = migrate_pickle_to_npz(pkl_path=pkl_path, npz_path=npz_path, json_path=json_path)
    assert success is True

    migrated = load_features_npz(npz_path=npz_path, json_path=json_path)
    assert len(migrated) == 3
    for k, v in original_dict.items():
        assert np.allclose(v, migrated[k], atol=1e-6)


# ── Test 5: Pending Submissions Workflow ──────────────────────────────────────

def test_pending_flow_does_not_affect_matching(tmp_path):
    """Verify that adding an image goes to pending and does NOT affect matching until confirmed."""
    pending_dir = str(tmp_path / "pending")
    reference_dir = str(tmp_path / "reference")
    trash_dir = str(tmp_path / "trash")
    Path(reference_dir).mkdir()

    # Step 1: Initialize matcher with known tiger
    matcher = TigerMatcher()
    known_emb = np.ones(128, dtype=np.float32)
    known_emb = known_emb / np.linalg.norm(known_emb)
    matcher.add_reference("tiger_known", known_emb, "dummy/path.jpg")

    # Step 2: Submit a new tiger image to pending
    test_img = Image.new("RGB", (64, 64), color=(100, 200, 50))
    success, msg, meta = submit_to_pending(
        image_input=test_img,
        tiger_name="tiger_candidate",
        provenance={"source": "camera_trap_1"},
        pending_dir=pending_dir
    )
    assert success is True
    sub_id = meta["submission_id"]

    # Verify pending state on disk
    pending_list = list_pending_submissions(pending_dir=pending_dir)
    assert len(pending_list) == 1
    assert pending_list[0]["submission_id"] == sub_id
    assert pending_list[0]["status"] == "pending"
    assert pending_list[0]["provenance"]["source"] == "camera_trap_1"

    # Step 3: CRITICAL CHECK — Verify matcher does NOT see pending tiger!
    query_emb = np.zeros(128, dtype=np.float32)
    query_emb[0] = 1.0
    stats = matcher.get_statistics()
    assert "tiger_candidate" not in stats["individuals"]
    assert stats["total_individuals"] == 1

    # Step 4: Review and approve the submission
    class MockExtractor:
        def extract_features(self, path):
            return np.ones(128, dtype=np.float32) / np.sqrt(128)

    mock_extractor = MockExtractor()
    ref_features_dict = {}

    ok, rev_msg, updated_meta = review_pending_submission(
        submission_id=sub_id,
        action="approve",
        reviewer_note="Verified by lead biologist",
        extractor=mock_extractor,
        matcher=matcher,
        reference_features=ref_features_dict,
        pending_dir=pending_dir,
        reference_dir=reference_dir,
        trash_dir=trash_dir,
        npz_path=str(tmp_path / "test_ref.npz"),
        json_path=str(tmp_path / "test_ref.json")
    )
    assert ok is True
    assert updated_meta["status"] == "approved"

    # Step 5: After reviewer confirmation, matcher now contains the new tiger!
    new_stats = matcher.get_statistics()
    assert "tiger_candidate" in new_stats["individuals"]
    assert new_stats["total_individuals"] == 2


def test_pending_flow_rejection(tmp_path):
    """Verify that rejecting a pending submission moves image to trash."""
    pending_dir = str(tmp_path / "pending")
    trash_dir = str(tmp_path / "trash")

    test_img = Image.new("RGB", (64, 64), color=(50, 50, 50))
    success, msg, meta = submit_to_pending(
        image_input=test_img,
        tiger_name="tiger_rejected",
        pending_dir=pending_dir
    )
    assert success is True
    sub_id = meta["submission_id"]

    ok, rev_msg, updated_meta = review_pending_submission(
        submission_id=sub_id,
        action="reject",
        reviewer_note="Blurry non-target animal",
        pending_dir=pending_dir,
        trash_dir=trash_dir
    )
    assert ok is True
    assert updated_meta["status"] == "rejected"
    assert Path(updated_meta["trash_path"]).exists()


# ── Test 6: Move to Trash Safety ──────────────────────────────────────────────

def test_move_to_trash(tmp_path):
    """Verify files/directories are moved to trash/ without rmtree."""
    trash_dir = str(tmp_path / "trash")
    sample_dir = tmp_path / "tiger_to_delete"
    sample_dir.mkdir()
    (sample_dir / "img.jpg").write_text("dummy", encoding="utf-8")

    trash_dest = move_to_trash(str(sample_dir), trash_dir=trash_dir)
    assert not sample_dir.exists()
    assert Path(trash_dest).exists()
    assert (Path(trash_dest) / "img.jpg").exists()


# ── Test 7: Terminology and Disclaimers ────────────────────────────────────────

def test_terminology_and_disclaimers():
    """Verify honesty rules: no 'Confidence', no 'MATCH FOUND', disclaimers present."""
    matcher = TigerMatcher()
    t1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    matcher.add_reference("tiger_001", t1, "data/ref/001.jpg")

    # High match test
    best_match, score, _ = matcher.find_match(t1, threshold=0.96)
    assert best_match == "tiger_001"
    assert score >= 0.96

    # Low match test
    orthogonal = np.zeros(128, dtype=np.float32)
    orthogonal[0] = 1.0
    best_match_low, score_low, _ = matcher.find_match(orthogonal, threshold=0.96)
    assert best_match_low is None

    # Verify top candidates structure (top-5)
    candidates = matcher.get_top_candidates(t1, top_k=5)
    assert len(candidates) == 1
    assert candidates[0]["name"] == "tiger_001"
    assert "reference_images" in candidates[0]
    assert "score" in candidates[0]

    # Verify terminology constants
    assert SIMILARITY_LABEL == "Cosine similarity"
    assert "Confidence" not in SIMILARITY_LABEL
    assert OUTCOME_ABOVE_THRESHOLD == "Candidate above threshold (unverified)"
    assert OUTCOME_BELOW_THRESHOLD == "No candidate above threshold"
    assert "Similarity indicates visual resemblance, not verified individual identity." in SIMILARITY_NOTE
