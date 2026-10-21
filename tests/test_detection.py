"""Tests for real detection and species verification stage prior to tiger re-identification:
1. No animal in frame -> no crops, re-ID does not run.
2. Person-only frame -> privacy enforcement, no person cropped or stored, re-ID does not run.
3. Non-tiger animal -> species named if verified, re-ID bypassed.
4. Tiger crop -> real classifier labels as tiger, re-ID forward pass executed.
5. Tadoba species list verification and dynamic capability check.
"""

import os
import json
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import (
    AnimalDetector,
    SpeciesClassifier,
    process_frame,
    get_tadoba_species_with_capabilities,
    TADOBA_SPECIES_JSON_PATH
)
from src.matcher import TigerMatcher


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    return AnimalDetector(device='cpu')


@pytest.fixture(scope="module")
def classifier():
    return SpeciesClassifier(device='cpu')


@pytest.fixture
def dummy_reid_system():
    matcher = TigerMatcher()
    t1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    matcher.add_reference("tiger_001", t1, "data/ref/001.jpg")

    extractor = MagicMock()
    extractor.extract_features.return_value = t1
    return extractor, matcher


# ── Test 1: No Animal in Frame ────────────────────────────────────────────────

def test_no_animal_in_frame(detector, classifier, dummy_reid_system):
    """Verify that a frame with no detected animals produces no crops and does not trigger re-ID."""
    extractor, matcher = dummy_reid_system

    # Solid grey image produces no detections
    blank_img = Image.new("RGB", (320, 320), color=(128, 128, 128))
    result = process_frame(
        image=blank_img,
        detector=detector,
        classifier=classifier,
        reid_extractor=extractor,
        reid_matcher=matcher
    )

    assert len(result['detections']) == 0
    assert len(result['tiger_crops']) == 0
    assert "No animals or objects detected" in result['summary_text']
    extractor.extract_features.assert_not_called()


# ── Test 2: Person-Only Frame (Privacy Enforcement) ───────────────────────────

def test_person_only_frame_privacy(classifier, dummy_reid_system):
    """Verify privacy rule: people are ignored, never cropped, stored, or re-identified."""
    extractor, matcher = dummy_reid_system

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{
        'box': [50.0, 50.0, 200.0, 300.0],
        'score': 0.88,
        'raw_label': 'person',
        'category': 'person'
    }]

    test_img = Image.new("RGB", (400, 400), color=(180, 180, 180))
    result = process_frame(
        image=test_img,
        detector=mock_detector,
        classifier=classifier,
        reid_extractor=extractor,
        reid_matcher=matcher
    )

    assert len(result['detections']) == 1
    assert result['detections'][0]['category'] == 'person'
    assert "Ignored per privacy policy" in result['detections'][0]['label']
    # CRITICAL: People must NEVER be cropped or stored
    assert len(result['tiger_crops']) == 0
    # Re-ID must NEVER run on a person
    extractor.extract_features.assert_not_called()


# ── Test 3: Non-Tiger Animal (No Re-ID) ────────────────────────────────────────

def test_non_tiger_animal_no_reid(dummy_reid_system):
    """Verify non-tiger animals are labeled with their species and bypassed from tiger re-ID."""
    extractor, matcher = dummy_reid_system

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{
        'box': [30.0, 30.0, 150.0, 150.0],
        'score': 0.82,
        'raw_label': 'bear',
        'category': 'animal'
    }]

    mock_classifier = MagicMock()
    mock_classifier.classify_crop.return_value = {
        'label': 'sloth bear',
        'score': 0.91,
        'is_tiger': False,
        'class_index': 297
    }

    test_img = Image.new("RGB", (300, 300), color=(40, 40, 40))
    result = process_frame(
        image=test_img,
        detector=mock_detector,
        classifier=mock_classifier,
        reid_extractor=extractor,
        reid_matcher=matcher
    )

    assert len(result['detections']) == 1
    det = result['detections'][0]
    assert det['is_tiger'] is False
    assert "sloth bear" in det['label']
    assert "Bypassed" in det['reid_result']
    assert len(result['tiger_crops']) == 0
    extractor.extract_features.assert_not_called()


def test_animal_detected_species_not_determined_when_no_classifier(dummy_reid_system):
    """Verify that when no species classifier is provided, it shows 'Animal detected (species not determined)'."""
    extractor, matcher = dummy_reid_system

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{
        'box': [20.0, 20.0, 100.0, 100.0],
        'score': 0.75,
        'raw_label': 'cat',
        'category': 'animal'
    }]

    test_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    result = process_frame(
        image=test_img,
        detector=mock_detector,
        classifier=None,  # No classifier loaded
        reid_extractor=extractor,
        reid_matcher=matcher
    )

    assert len(result['detections']) == 1
    det = result['detections'][0]
    assert det['label'] == "Animal detected (species not determined)"
    assert det['is_tiger'] is False
    assert len(result['tiger_crops']) == 0
    extractor.extract_features.assert_not_called()


# ── Test 4: Tiger Crop Triggers Re-ID ──────────────────────────────────────────

def test_tiger_crop_triggers_reid(dummy_reid_system):
    """Verify that only crops labeled as tiger by the classifier trigger tiger re-ID."""
    extractor, matcher = dummy_reid_system

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{
        'box': [10.0, 10.0, 200.0, 200.0],
        'score': 0.94,
        'raw_label': 'cat',
        'category': 'animal'
    }]

    mock_classifier = MagicMock()
    mock_classifier.classify_crop.return_value = {
        'label': 'tiger',
        'score': 0.97,
        'is_tiger': True,
        'class_index': 292
    }

    test_img = Image.new("RGB", (300, 300), color=(220, 130, 40))
    result = process_frame(
        image=test_img,
        detector=mock_detector,
        classifier=mock_classifier,
        reid_extractor=extractor,
        reid_matcher=matcher
    )

    assert len(result['detections']) == 1
    det = result['detections'][0]
    assert det['is_tiger'] is True
    assert "Tiger (Panthera tigris)" in det['label']
    assert len(result['tiger_crops']) == 1
    # Extractor MUST be called on the tiger crop
    extractor.extract_features.assert_called_once()
    assert "Top candidate: tiger_001" in det['reid_result']


# ── Test 5: Tadoba Species JSON Integrity & Model Capabilities ─────────────────

def test_tadoba_species_json_integrity(classifier):
    """Verify data/tadoba-species.json structure, sources, and dynamic capability mapping."""
    assert os.path.exists(TADOBA_SPECIES_JSON_PATH)

    with open(TADOBA_SPECIES_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Must be a list of species entries
    assert isinstance(data, list)
    assert len(data) > 0

    # Ensure NO total count key is stated
    assert not isinstance(data, dict) or "total_count" not in data

    allowed_sources = {
        "https://zsi.gov.in/",
        "https://ntca.gov.in/",
        "https://www.keybiodiversityareas.org/site/factsheet/18278",
        "https://mytadoba.mahaforest.gov.in/"
    }

    for item in data:
        assert "common_name" in item and item["common_name"]
        assert "scientific_name" in item and item["scientific_name"]
        assert "group" in item and item["group"] in ("Mammals", "Birds", "Reptiles", "Amphibians")
        assert "tier" in item and item["tier"]
        assert "source_urls" in item and isinstance(item["source_urls"], list)
        for url in item["source_urls"]:
            assert url in allowed_sources
        assert item["verification_status"] == "verified"

    # Compute model capabilities
    capabilities = get_tadoba_species_with_capabilities(classifier=classifier)
    assert len(capabilities) == len(data)

    tiger_entry = next(e for e in capabilities if e["common_name"] == "Bengal Tiger")
    assert tiger_entry["model_can_name"] is True
    assert tiger_entry["matched_model_label"] == "tiger"

    # Verify that species absent from ImageNet return model_can_name = False
    chital_entry = next(e for e in capabilities if "Chital" in e["common_name"])
    assert chital_entry["model_can_name"] is False


# ── Test 6: Bounded Count & Heuristic Score Labels ─────────────────────────────

def test_bounded_detections_and_score_interpretation(detector):
    """Verify that detections are bounded and labeled as heuristic scores, not probabilities."""
    test_img = Image.new("RGB", (320, 320))
    detections = detector.detect(test_img, max_detections=3)
    assert len(detections) <= 3


# ── Test 7: Webcam Coordinate Normalization & Error Separation Regression ──────

def test_webcam_box_coordinates_and_drawing_regression():
    """
    Regression test for the webcam photo bug:
    A detection on a non-320x320 frame (e.g. 640x480 webcam) with reversed or
    out-of-bounds coordinates must be normalized and clipped without raising
    PIL's 'ValueError: x1 must be greater than or equal to x0'.
    """
    mock_detector = MagicMock()
    # Detection with reversed x coordinates (x1 > x2) and coordinates extending beyond 640x480
    mock_detector.detect.return_value = [
        {
            'box': [500.0, 100.0, 200.0, 350.0],  # reversed x: 500 > 200
            'score': 0.85,
            'raw_label': 'person',
            'category': 'person'
        },
        {
            'box': [10.0, 450.0, 100.0, 50.0],   # reversed y: 450 > 50
            'score': 0.72,
            'raw_label': 'cat',
            'category': 'animal'
        },
        {
            'box': [50.0, 50.0, 50.0, 50.0],     # zero-area: skipped
            'score': 0.60,
            'raw_label': 'dog',
            'category': 'animal'
        }
    ]

    mock_classifier = MagicMock()
    mock_classifier.classify_crop.return_value = {
        'label': 'domestic cat',
        'score': 0.80,
        'is_tiger': False,
        'class_index': 282
    }

    webcam_frame = Image.new("RGB", (640, 480), color=(100, 120, 140))
    # Should execute without any ValueError
    result = process_frame(
        image=webcam_frame,
        detector=mock_detector,
        classifier=mock_classifier
    )

    assert result['annotated_image'] is not None
    assert result['annotated_image'].size == (640, 480)
    # Zero-area box was skipped, 2 valid boxes processed
    assert len(result['detections']) == 2
    # Verify coordinates were normalized so x1 < x2 and y1 < y2
    for det in result['detections']:
        bx1, by1, bx2, by2 = det['box']
        assert bx2 > bx1
        assert by2 > by1
        assert 0 <= bx1 <= 640 and 0 <= bx2 <= 640
        assert 0 <= by1 <= 480 and 0 <= by2 <= 480


def test_identify_tiger_separates_validation_from_internal_errors():
    """
    Verify that real validation failures produce friendly 'Image not accepted' messages,
    while internal ValueErrors during processing produce friendly internal error messages
    and are NEVER labeled as upload rejections.
    """
    import demo
    from unittest.mock import patch

    # Ensure database is loaded or mocked first
    mock_matcher = MagicMock()
    mock_matcher.reference_features = {"dummy": np.zeros(128)}
    demo.matcher = mock_matcher

    # 1. Genuine validation failure (invalid image type)
    report, img, html, gallery = demo.identify_tiger(None)
    assert "Please upload an image to identify." in report

    # Genuine validation failure with invalid file path
    report, img, html, gallery = demo.identify_tiger("non_existent_file.xyz")
    assert "Image not accepted" in report
    assert "Upload rejected" not in report

    # 2. Internal ValueError during pipeline execution
    valid_img = Image.new("RGB", (320, 320), color=(128, 128, 128))

    with patch("demo.process_frame", side_effect=ValueError("x1 must be greater than or equal to x0")):
        report, img, html, gallery = demo.identify_tiger(valid_img)
        # MUST NOT say "Image not accepted" or "Upload rejected"
        assert "Image not accepted" not in report
        assert "Upload rejected" not in report
        # MUST show friendly internal error message
        assert "Something went wrong while processing this image." in report

