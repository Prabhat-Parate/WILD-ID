import gradio as gr
import os
import sys
import time
import json
import csv
import io
import torch
from pathlib import Path
from PIL import Image

sys.path.append('src')

from src.config import (
    SIMILARITY_THRESHOLD,
    THRESHOLD_LABEL,
    TOP_K_CANDIDATES,
    SIMILARITY_LABEL,
    SIMILARITY_NOTE,
    OUTCOME_ABOVE_THRESHOLD,
    OUTCOME_BELOW_THRESHOLD,
    REFERENCE_DIR,
    PENDING_DIR,
    TRASH_DIR,
    TEST_DIR,
    NPZ_PATH,
    METADATA_JSON_PATH,
    LEGACY_PKL_PATH,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_UPLOAD_SIZE_MB,
    MAX_PIXEL_COUNT,
    MAX_DISPLAY_PIXELS,
    QUEUE_MAX_SIZE,
    DEFAULT_CONCURRENCY_LIMIT,
    TORCH_NUM_THREADS,
    REQUEST_TIMEOUT_SECONDS
)
from src.feature_extractor import TigerFeatureExtractor
from src.matcher import TigerMatcher
from src.speciesnet_adapter import SpeciesNetAdapter
from src.detector import (
    AnimalDetector,
    SpeciesClassifier,
    process_frame,
    get_tadoba_species_with_capabilities
)
from src.utils import (
    parse_reference_structure,
    save_features_npz,
    load_features,
    load_images_from_directory,
    validate_dataset,
    validate_identity_name,
    ensure_rgb_image,
    safe_process_image,
    ImageValidationError,
    run_with_timeout,
    move_to_trash,
    submit_to_pending,
    list_pending_submissions,
    review_pending_submission
)
from src.logger import logger, log_inference_event

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = str(BASE_DIR / "static" / "css" / "wildid.css")
JS_PATH = str(BASE_DIR / "static" / "js" / "wildid.js")
FAVICON_PATH = str(BASE_DIR / "static" / "favicon.svg")

# Global runtime state - single shared model instances
extractor = None
matcher = None
detector = None
classifier = None
speciesnet_classifier = None
reference_features = {}


def initialize_system():
    global extractor, matcher, detector, classifier, speciesnet_classifier
    torch.set_num_threads(TORCH_NUM_THREADS)

    if extractor is None:
        extractor = TigerFeatureExtractor(device='cpu')
    if matcher is None:
        matcher = TigerMatcher()
    if detector is None:
        detector = AnimalDetector(device='cpu')
    if classifier is None:
        classifier = SpeciesClassifier(device='cpu')

    # Disabled unless WILDID_SPECIESNET_ENABLED=1 is set before startup.
    if SpeciesNetAdapter.enabled() and speciesnet_classifier is None:
        speciesnet_classifier = SpeciesNetAdapter()

    logger.info(
        f"System initialized: PyTorch threads={torch.get_num_threads()}, "
        f"device=CPU, shared model instances loaded."
    )


# ── HTML Helpers ──────────────────────────────────────────────────────────────

def _esc(text):
    """Minimal HTML escaping for dynamic values."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def get_page_header_html(title, desc, action_label=None, action_page_idx=None):
    """Generate uniform page header across all pages."""
    action_btn = ""
    if action_label and action_page_idx is not None:
        action_btn = (
            f'<div class="wid-page-header-action">'
            f'<button class="wid-btn-header" onclick="window.__wid_switch_page({action_page_idx})">'
            f'{_esc(action_label)}'
            f'</button>'
            f'</div>'
        )
    return (
        f'<div class="wid-page-header">'
        f'  <div class="wid-page-header-text">'
        f'    <h1 class="wid-page-title">{_esc(title)}</h1>'
        f'    <p class="wid-page-desc">{_esc(desc)}</p>'
        f'  </div>'
        f'  {action_btn}'
        f'</div>'
    )


def get_overview_status_html():
    """Return key-value status card populated from real runtime objects."""
    # 1. Model status & loaded names
    if extractor is not None:
        model_status_badge = '<span class="wid-status-badge-inline"><span class="wid-dot-green"></span> Loaded</span>'
        reid_name = "ViT-Base (vit_base_patch16_224, 128-d)"
    else:
        model_status_badge = '<span class="wid-status-badge-inline"><span class="wid-dot-amber"></span> Unloaded</span>'
        reid_name = "None"

    detector_name = "SSDLite320-MobileNetV3 (COCO)" if detector is not None else "Unloaded"
    species_name = "MobileNetV3-Small (ImageNet-1k)" if classifier is not None else "Unloaded"

    # 2. Database individuals & reference images as separate counts
    if matcher is not None and matcher.reference_features:
        stats = matcher.get_statistics()
        db_badge = '<span class="wid-status-badge-inline"><span class="wid-dot-green"></span> Loaded</span>'
        num_indiv = str(stats["total_individuals"])
        num_images = str(stats["total_images"])
    else:
        db_badge = '<span class="wid-status-badge-inline"><span class="wid-dot-amber"></span> No database loaded</span>'
        num_indiv = "0"
        num_images = "0"

    threshold_val = f'{SIMILARITY_THRESHOLD:.2f} <em class="wid-tag-amber">{_esc(THRESHOLD_LABEL)}</em>'

    return (
        f'<div class="wid-status-panel">'
        f'  <div class="wid-status-panel-title">System & Database Status</div>'
        f'  <dl class="wid-status-dl">'
        f'    <div class="wid-status-row"><dt>Embedding Model</dt><dd>{model_status_badge}</dd></div>'
        f'    <div class="wid-status-row"><dt>Re-ID Backbone</dt><dd>{_esc(reid_name)}</dd></div>'
        f'    <div class="wid-status-row"><dt>Localizer & Classifier</dt><dd>{_esc(detector_name)} &bull; {_esc(species_name)}</dd></div>'
        f'    <div class="wid-status-row"><dt>Reference Database</dt><dd>{db_badge}</dd></div>'
        f'    <div class="wid-status-row"><dt>Known Individuals</dt><dd>{num_indiv}</dd></div>'
        f'    <div class="wid-status-row"><dt>Reference Images</dt><dd>{num_images}</dd></div>'
        f'    <div class="wid-status-row"><dt>Threshold</dt><dd>{threshold_val}</dd></div>'
        f'  </dl>'
        f'</div>'
    )


def get_pipeline_steps_html():
    """Return horizontal step diagram of the four real pipeline stages."""
    steps = [
        ("01", "Localization", "SSDLite320-MobileNetV3", "Locates candidate animal boxes; bounded detection output."),
        ("02", "Privacy Filter", "Built-in Rule", "Detects and suppresses human presence; never cropped or stored."),
        ("03", "Species Verifier", "MobileNetV3-Small", "Classifies crops; non-tiger fauna bypassed from re-ID."),
        ("04", "Re-Identification", "ViT-Base (128-dim)", "Cosine ranking against reference vector store.")
    ]

    parts = [
        '<div class="wid-pipeline-diagram">',
        '  <div class="wid-diagram-title">Four-Stage Inference Pipeline</div>',
        '  <div class="wid-steps-track">'
    ]
    for idx, (num, name, model, desc) in enumerate(steps):
        parts.append(
            f'    <div class="wid-step-card">'
            f'      <div class="wid-step-num">STAGE {num}</div>'
            f'      <div class="wid-step-name">{_esc(name)}</div>'
            f'      <div class="wid-step-model">{_esc(model)}</div>'
            f'      <div class="wid-step-desc">{_esc(desc)}</div>'
            f'    </div>'
        )
        if idx < len(steps) - 1:
            parts.append('    <div class="wid-step-arrow">&rarr;</div>')
    parts.append('  </div>')
    parts.append('</div>')
    return "\n".join(parts)


def get_db_status_html():
    """Return a small HTML fragment showing current database and model status."""
    if matcher is not None and matcher.reference_features:
        stats = matcher.get_statistics()
        db_part = (
            f'<span class="wid-status-loaded">Database loaded</span>'
            f' &mdash; {stats["total_individuals"]} individuals, '
            f'{stats["total_images"]} reference images'
        )
    else:
        db_part = '<span class="wid-status-unloaded">No database loaded</span>'

    model_part = (
        '<span class="wid-status-loaded">Model loaded</span>'
        if extractor is not None
        else '<span class="wid-status-unloaded">Model not loaded</span>'
    )

    return (
        f'<div id="wid-header">'
        f'<p class="wid-wordmark">WILD-ID</p>'
        f'<p class="wid-subtitle">Individual tiger re-identification</p>'
        f'<p class="wid-status">{db_part} &nbsp;|&nbsp; {model_part}</p>'
        f'</div>'
    )


def build_reid_html(best_match, best_score, top_candidates, threshold):
    """Build a structured HTML results panel for re-identification ranking."""
    is_above = best_match is not None
    outcome_text = OUTCOME_ABOVE_THRESHOLD if is_above else OUTCOME_BELOW_THRESHOLD
    outcome_class = "above" if is_above else "below"

    # 1. Summary line first
    parts = [
        f'<div class="wid-identify-summary-line {outcome_class}">'
        f'  <strong>Outcome:</strong> {_esc(outcome_text)} &mdash; '
        f'  Highest cosine: <code>{best_score:.4f}</code> '
        f'  (Threshold: <code>{threshold:.4f}</code> &bull; {_esc(THRESHOLD_LABEL)})'
        f'</div>',
        f'<div class="wid-candidate-list">'
    ]

    # 2. Top 5 candidates with file name and numeric cosine beside a thin bar
    for i, cand in enumerate(top_candidates[:5], 1):
        name = cand["name"]
        score = cand["score"]
        bar_pct = max(0.0, min(100.0, score * 100.0))
        ref_files = cand.get("reference_images", [])
        ref_basename = os.path.basename(ref_files[0]) if ref_files else "reference_crop.jpg"
        bar_cls = "above" if score >= threshold else "below"

        parts.append(
            f'  <div class="wid-candidate-row">'
            f'    <div class="wid-cand-header">'
            f'      <span class="wid-cand-name">#{i} {_esc(name)}</span>'
            f'      <span class="wid-cand-file">{_esc(ref_basename)}</span>'
            f'    </div>'
            f'    <div class="wid-cand-score-line">'
            f'      <span class="wid-cand-score-val">{score:.4f}</span>'
            f'      <div class="wid-cand-bar-track">'
            f'        <div class="wid-cand-bar-fill {bar_cls}" style="width:{bar_pct:.1f}%;"></div>'
            f'      </div>'
            f'    </div>'
            f'  </div>'
        )

    parts.append('</div>')
    return "\n".join(parts)


def build_batch_html(rows, total_count):
    """Build an HTML table for batch results."""
    parts = [
        f'<div class="wid-result-summary below-threshold">',
        f'  <div class="wid-summary-outcome">Batch ranking complete &mdash; {total_count} images</div>',
        f'  <div class="wid-summary-detail">',
        f'    Threshold: {SIMILARITY_THRESHOLD:.4f} ({_esc(THRESHOLD_LABEL)})',
        f'    &nbsp;|&nbsp; {_esc(SIMILARITY_NOTE)}',
        f'  </div>',
        f'</div>',
        f'<div class="wid-table-scroll">',
        f'<table class="wid-results-table">',
        f'<thead><tr>',
        f'  <th>File</th><th>Best candidate</th><th>{_esc(SIMILARITY_LABEL)}</th><th></th><th>Outcome</th>',
        f'</tr></thead>',
        f'<tbody>',
    ]

    for idx, row in enumerate(rows, 1):
        bar_pct = max(0, min(100, row["score"] * 100))
        bar_class = "above-threshold" if row["above"] else "below-threshold"
        row_class = "wid-outcome-above" if row["above"] else "wid-outcome-below"

        stagger_idx = min(idx, 10)
        parts.append(f'<tr style="--row-index:{stagger_idx};">')
        parts.append(f'  <td>{_esc(row["file"])}</td>')
        parts.append(f'  <td class="{row_class}">{_esc(row["candidate"])}</td>')
        parts.append(f'  <td style="font-variant-numeric:tabular-nums;">{row["score"]:.4f}</td>')
        parts.append(
            f'  <td><span class="wid-sim-bar-bg">'
            f'<span class="wid-sim-bar-fill {bar_class}" style="width:{bar_pct:.1f}%"></span>'
            f'</span></td>'
        )
        parts.append(f'  <td class="{row_class}">{_esc(row["outcome"])}</td>')
        parts.append(f'</tr>')

    parts.append(f'</tbody></table>')
    parts.append(f'</div>')

    return "\n".join(parts)


# ── Database Management ───────────────────────────────────────────────────────

def load_existing_database():
    """Load reference database from .npz and JSON into matcher."""
    global extractor, matcher, reference_features

    if extractor is None:
        initialize_system()

    if not os.path.exists(NPZ_PATH) and not os.path.exists(LEGACY_PKL_PATH):
        return (
            f"No existing database found at:\n  {NPZ_PATH}\n\n"
            "Use 'Build Database from Images' or run 'python migrate_database.py' first."
        ), get_db_status_html()

    reference_features = load_features()

    if not reference_features:
        return "Database file exists but is empty. Rebuild it.", get_db_status_html()

    matcher = TigerMatcher()
    for img_path, features in reference_features.items():
        parts = Path(img_path).parts
        try:
            name = parts[-2]
        except IndexError:
            name = "unknown"
        matcher.add_reference(name, features, img_path)

    stats = matcher.get_statistics()
    result = (
        f"Database loaded successfully.\n\n"
        f"Individuals   : {stats['total_individuals']}\n"
        f"Total images  : {stats['total_images']}\n\n"
        f"Known tigers ({stats['total_individuals']}):\n" +
        "\n".join(f"  {name}" for name in sorted(stats['individuals']))
    )
    return result, get_db_status_html()


def build_reference_database():
    """Scan data/reference/, extract embeddings, atomically save NPZ + JSON."""
    global extractor, matcher, reference_features

    if extractor is None:
        initialize_system()

    matcher = TigerMatcher()
    reference_features = {}

    individuals = parse_reference_structure(REFERENCE_DIR)
    if not individuals:
        yield "No reference images found. Add images to data/reference/", get_db_status_html()
        return

    all_tasks = []
    for name, image_paths in individuals.items():
        for img_path in image_paths:
            all_tasks.append((name, img_path))

    total = len(all_tasks)
    build_start = time.time()

    def fmt_time(seconds):
        seconds = max(0, int(seconds))
        if seconds >= 60:
            m, s = divmod(seconds, 60)
            return f"{m}m {s:02d}s"
        return f"{seconds}s"

    for idx, (name, img_path) in enumerate(all_tasks):
        elapsed = time.time() - build_start
        done = idx

        if done > 0:
            avg_per_image = elapsed / done
            remaining_est = avg_per_image * (total - done)
            remaining_str = fmt_time(remaining_est)
        else:
            remaining_str = "calculating..."

        live_status = (
            f"Building database...\n"
            f"{'_'*40}\n\n"
            f"Processing : {os.path.basename(img_path)}\n"
            f"Progress   : {done}/{total} images done\n\n"
            f"Elapsed    : {fmt_time(elapsed)}\n"
            f"Remaining  : {remaining_str}"
        )
        yield live_status, get_db_status_html()

        features = extractor.extract_features(img_path)
        if features is not None:
            matcher.add_reference(name, features, img_path)
            reference_features[img_path] = features

    save_features_npz(reference_features, NPZ_PATH, METADATA_JSON_PATH)

    total_elapsed = time.time() - build_start
    stats = matcher.get_statistics()

    result = (
        f"Database built and saved.\n"
        f"{'_'*40}\n\n"
        f"Individuals loaded : {stats['total_individuals']}\n"
        f"Total images       : {stats['total_images']}\n"
        f"Total time taken   : {fmt_time(total_elapsed)}\n\n"
        f"Known tigers ({stats['total_individuals']}):\n" +
        "\n".join(f"  {name}" for name in sorted(stats['individuals']))
    )
    yield result, get_db_status_html()


def delete_tiger(tiger_name, confirm_delete=False):
    """Move a tiger to trash/ after explicit confirmation."""
    global matcher, reference_features

    tiger_name = (tiger_name or "").strip()
    if not tiger_name:
        return "Please enter a tiger name to delete."

    if not confirm_delete:
        return (
            "Deletion aborted: explicit confirmation required.\n"
            "Check the 'Confirm moving tiger to trash' box before proceeding."
        )

    is_valid, sanitized_name = validate_identity_name(tiger_name, REFERENCE_DIR)
    if not is_valid:
        return f"Invalid identity name: {sanitized_name}"

    if matcher is None:
        return "Database not loaded. Please load or build the database first."

    if sanitized_name not in matcher.reference_features:
        known = sorted(matcher.reference_features.keys())
        return (
            f"'{sanitized_name}' not found in database.\n\n"
            f"Known tigers ({len(known)} total):\n" + "\n".join(f"  {n}" for n in known[:20])
        )

    del matcher.reference_features[sanitized_name]
    if sanitized_name in matcher.reference_names:
        del matcher.reference_names[sanitized_name]

    keys_to_remove = [k for k in reference_features if Path(k).parts[-2] == sanitized_name]
    for k in keys_to_remove:
        del reference_features[k]

    save_features_npz(reference_features, NPZ_PATH, METADATA_JSON_PATH)

    tiger_folder = Path(REFERENCE_DIR) / sanitized_name
    trash_dest = None
    if tiger_folder.exists():
        trash_dest = move_to_trash(str(tiger_folder), TRASH_DIR)

    stats = matcher.get_statistics()
    return (
        f"Tiger successfully removed.\n\n"
        f"Tiger name     : {sanitized_name}\n"
        f"Files moved to : {trash_dest if trash_dest else 'no files found on disk'}\n"
        f"Entries removed: {len(keys_to_remove)}\n\n"
        f"Database now   : {stats['total_individuals']} individuals, "
        f"{stats['total_images']} reference images."
    )


# ── Detection & Identification Pipeline ───────────────────────────────────────

def identify_tiger(image):
    """
    Run detection, privacy filtering, species verification,
    and gated individual re-identification with safety bounds and timeouts.

    Returns: (detection_report_text, annotated_image, reid_html, gallery_items)
    """
    global extractor, matcher, detector, classifier

    empty_html = '<div class="wid-empty-state"><strong>No results yet.</strong> Upload an image and press Identify.</div>'

    if matcher is None or not matcher.reference_features:
        return "No database loaded. Use Setup > Load Existing Database.", None, empty_html, []

    if image is None:
        return "Please upload an image to identify.", None, empty_html, []

    start_time = time.time()
    # Stage 1: Validate and ingest uploaded or webcam image
    try:
        rgb_image, img_meta = safe_process_image(image)
    except (ImageValidationError, ValueError) as ve:
        logger.warning(f"Upload rejected by safety guard: {ve}")
        return (
            f"Image not accepted: {str(ve)}\n\n"
            "Please check the file format (JPEG, PNG, WEBP), file size (under 15 MB), "
            "and resolution, then try again.",
            None, empty_html, []
        )
    except Exception as e:
        logger.warning(f"Failed to read image input: {e}")
        return (
            f"Image not accepted: Unable to read the uploaded image ({str(e)}).\n\n"
            "Please check that the file is a valid, uncorrupted image.",
            None, empty_html, []
        )

    # Stage 2: Inference pipeline execution
    try:
        # Log request without image content
        log_inference_event(
            action="IDENTIFY_TIGER_START",
            image_metadata=img_meta,
            status="STARTED"
        )

        if detector is None or classifier is None:
            initialize_system()

        # Execute frame pipeline under timeout
        def _execute_pipeline():
            return process_frame(
                image=rgb_image,
                detector=detector,
                classifier=classifier,
                reid_extractor=extractor,
                reid_matcher=matcher,
                score_threshold=0.25,
                max_detections=10,
                speciesnet_classifier=speciesnet_classifier,
                speciesnet_threshold=0.65
            )

        frame_res = run_with_timeout(_execute_pipeline, timeout=REQUEST_TIMEOUT_SECONDS)

        annotated_img = frame_res['annotated_image']
        detections = frame_res['detections']
        summary_text = frame_res['summary_text']
        tiger_crops = frame_res['tiger_crops']

        gallery_items = []
        top_cand_name = None
        top_cand_score = 0.0
        reid_html = '<div class="wid-empty-state"><strong>No tiger detected.</strong> Re-identification requires a verified tiger crop.</div>'

        # If any verified tiger was found, run full top-5 candidate retrieval
        tiger_detections = [d for d in detections if d.get('is_tiger')]
        if tiger_detections and tiger_crops:
            primary_crop = tiger_crops[0]
            features = extractor.extract_features(primary_crop)

            if features is not None:
                best_match, best_score, all_scores = matcher.find_match(features, threshold=SIMILARITY_THRESHOLD)
                top_candidates = matcher.get_top_candidates(features, top_k=TOP_K_CANDIDATES)
                top_cand_name = best_match
                top_cand_score = best_score

                # Build structured HTML result panel
                reid_html = build_reid_html(best_match, best_score, top_candidates, SIMILARITY_THRESHOLD)

                # Build gallery items from top candidate reference images
                for i, cand in enumerate(top_candidates, 1):
                    name = cand["name"]
                    score = cand["score"]
                    for ref_img in cand["reference_images"][:2]:
                        if os.path.exists(ref_img):
                            gallery_items.append((ref_img, f"#{i} {name} ({SIMILARITY_LABEL}: {score:.4f})"))

        # Build pipeline status strip (done, skipped, failed)
        loc_status = "done" if detections else ("done" if not detections else "failed")
        privacy_status = "done"  # Privacy check always executed
        
        has_animal = any(d.get('category') == 'animal' for d in detections)
        has_tiger = any(d.get('is_tiger') for d in detections)
        
        if not detections:
            spec_status = "skipped"
            reid_status = "skipped"
        elif not has_animal:
            spec_status = "skipped"
            reid_status = "skipped"
        elif not has_tiger:
            spec_status = "done"
            reid_status = "skipped"
        else:
            spec_status = "done"
            reid_status = "done" if (tiger_crops and top_cand_name is not None) else "failed"

        strip_html = (
            f'<div class="wid-pipeline-strip">'
            f'  <span class="wid-strip-label">Pipeline:</span>'
            f'  <span class="wid-strip-tag {loc_status}">1. Localize: {loc_status}</span>'
            f'  <span class="wid-strip-tag {privacy_status}">2. Privacy: {privacy_status}</span>'
            f'  <span class="wid-strip-tag {spec_status}">3. Species: {spec_status}</span>'
            f'  <span class="wid-strip-tag {reid_status}">4. Re-ID: {reid_status}</span>'
            f'</div>'
        )

        full_results_html = f"{strip_html}\n{reid_html}"

        elapsed_ms = (time.time() - start_time) * 1000
        log_inference_event(
            action="IDENTIFY_TIGER_COMPLETE",
            image_metadata=img_meta,
            status="SUCCESS",
            duration_ms=elapsed_ms,
            detail=f"detections={len(detections)}, top_cand={top_cand_name}, sim={top_cand_score:.4f}"
        )
        return summary_text, annotated_img, full_results_html, gallery_items

    except TimeoutError as te:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(f"Inference timed out after {elapsed_ms:.1f}ms: {te}")
        return (
            f"Request timed out: Inference exceeded {REQUEST_TIMEOUT_SECONDS}s limit. "
            "Please try a smaller image or retry later.",
            None, empty_html, []
        )

    except Exception as e:
        # Internal error during inference: log full traceback, show friendly non-technical message
        import traceback as _tb
        logger.error(f"Internal error during inference pipeline: {e}\n{_tb.format_exc()}")
        return (
            "Something went wrong while processing this image. "
            "Please try again, or try a different image.\n\n"
            "If this keeps happening, check the server logs for details.",
            None, empty_html, []
        )


# Backward compatibility alias
identify_leopard = identify_tiger


def add_new_tiger(tiger_name, image, provenance_note=""):
    """Submit a query image to pending set with provenance metadata."""
    if image is None:
        return "No query image available to submit."

    tiger_name = (tiger_name or "").strip()
    if not tiger_name:
        return "Please enter a tiger name / identifier."

    provenance = {
        "source": "web_ui_submission",
        "note": provenance_note.strip() if provenance_note else "Submitted via Gradio UI",
        "submitted_at_epoch": time.time()
    }

    success, msg, meta = submit_to_pending(
        image_input=image,
        tiger_name=tiger_name,
        provenance=provenance,
        pending_dir=PENDING_DIR
    )
    return msg


def get_pending_summary():
    submissions = list_pending_submissions(PENDING_DIR)
    if not submissions:
        return "No pending submissions currently waiting for review."

    lines = [f"Found {len(submissions)} pending submission(s):\n"]
    for s in submissions:
        sub_id = s.get("submission_id")
        t_name = s.get("tiger_name")
        t_date = s.get("created_at", "unknown")
        lines.append(f"  ID: {sub_id} | Name: {t_name} | Date: {t_date}")
    return "\n".join(lines)


def handle_review_submission(submission_id, action, reviewer_note=""):
    global extractor, matcher, reference_features

    submission_id = (submission_id or "").strip()
    if not submission_id:
        return "Please enter a Submission ID to review."

    success, msg, meta = review_pending_submission(
        submission_id=submission_id,
        action=action,
        reviewer_note=reviewer_note,
        extractor=extractor,
        matcher=matcher,
        reference_features=reference_features,
        pending_dir=PENDING_DIR,
        reference_dir=REFERENCE_DIR,
        trash_dir=TRASH_DIR
    )
    return msg


def get_tadoba_species_table():
    """Retrieve formatted Tadoba species list with model capabilities."""
    global classifier
    if classifier is None:
        initialize_system()

    species_list = get_tadoba_species_with_capabilities(classifier=classifier)
    if not species_list:
        return "No species records found in data/tadoba-species.json."

    lines = [
        "VERIFIED TADOBA-ANDHARI FAUNA & MODEL CAPABILITY",
        "=" * 70,
        "Verified against: ZSI Conservation Area Series (Vol 25), NTCA brief,",
        "BirdLife/KBA factsheet 18278, and Maharashtra Forest Dept checklist.",
        "Scores and capabilities computed directly from loaded model label set.",
        "=" * 70,
        f"{'Common Name':28s} {'Scientific Name':26s} {'Model Can Name?':16s} {'Matched Label':18s}",
        "-" * 90
    ]

    for s in species_list:
        c_name = s.get('common_name', '')[:27]
        s_name = s.get('scientific_name', '')[:25]
        can_name = "YES [Verified]" if s.get('model_can_name') else "NO (Bypassed)"
        label = s.get('matched_model_label', '')[:17]
        lines.append(f"{c_name:28s} {s_name:26s} {can_name:16s} {label:18s}")

    return "\n".join(lines)


# ── Batch Identification ──────────────────────────────────────────────────────

def batch_identify(test_dir=TEST_DIR):
    """Run batch ranking and return (html_results, csv_file_path)."""
    global extractor, matcher

    if matcher is None or not matcher.reference_features:
        empty = '<div class="wid-empty-state"><strong>No database loaded.</strong> Use Setup > Load Existing Database first.</div>'
        return empty, None

    test_images = load_images_from_directory(test_dir)
    if not test_images:
        empty = f'<div class="wid-empty-state"><strong>No test images found.</strong> Add images to {_esc(test_dir)}</div>'
        return empty, None

    rows = []
    for img_path in test_images:
        features = extractor.extract_features(img_path)
        if features is not None:
            best_match, sim_score, _ = matcher.find_match(features, threshold=SIMILARITY_THRESHOLD)
            if best_match is not None:
                outcome = OUTCOME_ABOVE_THRESHOLD
                candidate = best_match
            else:
                top_cands = matcher.get_top_matches(features, top_k=1)
                candidate = top_cands[0][0] if top_cands else "None"
                outcome = OUTCOME_BELOW_THRESHOLD

            rows.append({
                "file": os.path.basename(img_path),
                "candidate": candidate,
                "score": sim_score,
                "outcome": outcome,
                "above": best_match is not None,
            })

    html = build_batch_html(rows, len(test_images))

    # Generate CSV for download
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["file", "best_candidate", "cosine_similarity", "outcome"])
    for row in rows:
        writer.writerow([row["file"], row["candidate"], f"{row['score']:.4f}", row["outcome"]])

    csv_path = str(BASE_DIR / "batch_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_buffer.getvalue())

    return html, csv_path


# ── Gradio Interface ──────────────────────────────────────────────────────────

def get_sidebar_html():
    """Return the full sidebar + mobile-header HTML shell.

    The sidebar contains the WILD-ID wordmark (with aria-label for the
    JS letter-by-letter reveal), a subtitle, a live-system status dot,
    9 navigation buttons with inline SVG line icons and edge-bar indicator,
    and an honesty note at the bottom.

    No pill shapes, no fake metrics, no animation toward invented numbers.
    Plain-text noscript fallback is included via aria-label on .wid-wordmark.
    """
    nav_pages = [
        # (index, label, title, svg-path-d)
        (0, "Overview",           "Overview",
         "M3 12L12 3l9 9M5 10v9a1 1 0 001 1h4v-4h4v4h4a1 1 0 001-1V10"),
        (1, "Identify",           "Identify",
         "M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16"),
        (2, "Batch",              "Batch",
         "M4 6h16M4 10h16M4 14h10"),
        (3, "Reference Gallery",  "Reference Gallery",
         "M4 16l4-4 4 4 4-6 4 6M2 20h20M2 4h20"),
        (4, "Tadoba Species",     "Tadoba Species",
         "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"),
        (5, "Review Queue",       "Review Queue",
         "M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"),
        (6, "Evaluation",         "Evaluation",
         "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"),
        (7, "Database",           "Database",
         "M4 7a4 4 0 018 0c0 2.21-1.79 4-4 4S4 9.21 4 7zm0 0c0 2.21 1.79 4 4 4 4.42 0 8-1.79 8-4M4 7v10M20 7v10M4 17c0 2.21 3.58 4 8 4s8-1.79 8-4M4 12c0 2.21 3.58 4 8 4s8-1.79 8-4"),
        (8, "About",              "About",
         "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"),
    ]

    nav_items_html = ""
    for idx, label, title, path_d in nav_pages:
        active_class = " active" if idx == 0 else ""
        nav_items_html += (
            f'<button class="wid-nav-item{active_class}" '
            f'data-tab-index="{idx}" '
            f'aria-label="{label}" '
            f'title="{title}">'
            f'<span class="wid-nav-icon" aria-hidden="true">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{path_d}"/></svg>'
            f'</span>'
            f'<span class="wid-nav-label">{label}</span>'
            f'</button>\n'
        )

    return (
        # ── Mobile Header (hidden on desktop/tablet, shown on phone) ──────────
        '<div id="wid-mobile-header" role="banner">'
        '  <div class="wid-mobile-bar">'
        '    <button class="wid-hamburger-btn" id="wid-hamburger-btn" '
        '      aria-label="Open navigation menu" aria-expanded="false">'
        '      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        '        stroke="currentColor" stroke-width="2" stroke-linecap="round">'
        '        <line x1="3" y1="6" x2="21" y2="6"/>'
        '        <line x1="3" y1="12" x2="21" y2="12"/>'
        '        <line x1="3" y1="18" x2="21" y2="18"/>'
        '      </svg>'
        '    </button>'
        '    <div class="wid-mobile-brand">'
        '      <span class="wid-mobile-wordmark">WILD-ID</span>'
        '      <span class="wid-mobile-sep">/</span>'
        '      <span class="wid-mobile-page-title" id="wid-mobile-page-title">Overview</span>'
        '    </div>'
        '  </div>'
        '</div>'

        # ── Drawer Backdrop (mobile only) ─────────────────────────────────────
        '<div id="wid-drawer-backdrop"></div>'

        # ── Left 232px Sidebar (self-contained; no open div trick) ───────────
        '<nav id="wid-sidebar" aria-label="WILD-ID navigation">'

        # Brand
        '  <div class="wid-brand-wrap">'
        '    <p class="wid-wordmark" aria-label="WILD-ID">WILD-ID</p>'
        '    <p class="wid-subtitle">Individual tiger re-identification</p>'
        '    <div class="wid-status-badge">'
        '      <span class="wid-status-dot" aria-hidden="true"></span>'
        '      <span id="wid-status-text"></span>'
        '    </div>'
        '  </div>'

        # Navigation list
        f'  <div class="wid-nav-list" role="list">'
        f'    <div class="wid-nav-indicator" aria-hidden="true"></div>'
        f'    {nav_items_html}'
        f'  </div>'

        # Honesty footer
        '  <div class="wid-honesty-sidebar">'
        '    <strong>Note:</strong> Cosine similarity is a ranking signal, '
        '    not a probability, confidence, or verified identity.'
        '  </div>'

        '</nav>'
    )


with gr.Blocks(
    title="WILD-ID | Tiger Re-identification",
    fill_width=True,
) as demo:

    # ── Shell: true flex-row using gr.Row so Gradio places children side by side
    # scale=0 + min_width=232 on sidebar → fixed 232 px, no growth
    # scale=1 + min_width=0 on content   → takes remaining width, never wraps
    with gr.Row(elem_id="wid-shell", equal_height=False):

        # ── Left 232px Sidebar column ──────────────────────────────────────────
        with gr.Column(scale=0, min_width=232, elem_id="wid-sidebar-col"):
            gr.HTML(value=get_sidebar_html(), elem_id="wid-shell-top")

        # ── Content Pane column ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=0, elem_id="wid-content-pane"):

            header_html = gr.HTML(
                value=get_db_status_html(),
                elem_id="wid-header",
                visible=False,
            )

            # ── All 9 pages inside gr.Tabs (tab-nav hidden via CSS) ─────────
            with gr.Tabs(elem_id="wid-main-tabs"):

                # ── 0: Overview ──────────────────────────────────────────────────
                with gr.Tab("Overview", elem_id="wid-tab-overview"):
                    gr.HTML(
                        '<div class="wid-overview-grid">'
                        '  <div class="wid-overview-card">'
                        '    <div class="wid-card-label">Pipeline</div>'
                        '    <div class="wid-card-value">4-stage</div>'
                        '    <div class="wid-card-sub">Localize → Privacy → Species → Re-ID</div>'
                        '  </div>'
                        '  <div class="wid-overview-card">'
                        '    <div class="wid-card-label">Similarity metric</div>'
                        '    <div class="wid-card-value">Cosine</div>'
                        '    <div class="wid-card-sub">Ranking signal only — not a probability</div>'
                        '  </div>'
                        '  <div class="wid-overview-card">'
                        '    <div class="wid-card-label">Re-ID backbone</div>'
                        '    <div class="wid-card-value">ViT-Base</div>'
                        '    <div class="wid-card-sub">128-dim embedding, patch16_224</div>'
                        '  </div>'
                        '  <div class="wid-overview-card">'
                        '    <div class="wid-card-label">Localizer</div>'
                        '    <div class="wid-card-value">SSDLite320</div>'
                        '    <div class="wid-card-sub">MobileNetV3 backbone — COCO</div>'
                        '  </div>'
                        '</div>'
                    )
                    gr.HTML(
                        f'<div class="wid-note">'
                        f'<strong>Honesty reminder:</strong> '
                        f'{_esc(SIMILARITY_NOTE)} '
                        f'Threshold is currently set to <code>{SIMILARITY_THRESHOLD:.2f}</code> '
                        f'({_esc(THRESHOLD_LABEL)}). Results require expert review before any '
                        f'conservation decision.'
                        f'</div>'
                    )
                    gr.Markdown(
                        "#### Pipeline summary\n\n"
                        "1. **Localizer** — SSDLite320 finds animal bounding boxes. "
                        "People detections are suppressed and never stored.\n"
                        "2. **Species verifier** — MobileNetV3-Small classifies the crop. "
                        "Non-tiger species are skipped.\n"
                        "3. **Re-ID embedding** — ViT-Base maps the tiger crop to a "
                        "128-dimensional unit vector.\n"
                        "4. **Ranking** — Cosine similarity against the reference database "
                        "produces a ranked candidate list.\n\n"
                        "Use the **Identify** page for single images, **Batch** for the full "
                        "test directory, and **Database** to load or rebuild the reference store."
                    )

                # ── 1: Identify ──────────────────────────────────────────────────
                with gr.Tab("Identify", elem_id="wid-tab-identify"):
                    gr.Markdown("#### Detection and candidate ranking")
                    gr.HTML(
                        f'<div class="wid-note">'
                        f'<strong>Pipeline:</strong> '
                        f'Localizer (SSDLite320) &rarr; Privacy check (people suppressed) &rarr; '
                        f'Species verifier (MobileNetV3) &rarr; Tiger re-ID (ViT-Base).<br>'
                        f'<strong>{_esc(SIMILARITY_LABEL)} threshold:</strong> '
                        f'{SIMILARITY_THRESHOLD:.2f} ({_esc(THRESHOLD_LABEL)}). '
                        f'Detection scores are ranking heuristics, not probabilities.<br>'
                        f'<em>{_esc(SIMILARITY_NOTE)}</em>'
                        f'</div>'
                    )

                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["wid-query-col"]):
                            input_image = gr.Image(
                                label="Camera trap image",
                                type="pil",
                                elem_id="wid-query-image",
                            )
                            identify_btn = gr.Button(
                                "Identify",
                                variant="primary",
                                elem_id="wid-identify-btn",
                            )
                            detection_annotated_img = gr.Image(
                                label="Detections (boxes and heuristic scores)",
                                type="pil",
                                elem_id="wid-annotated-image",
                            )

                        with gr.Column(scale=1, elem_classes=["wid-results-col"]):
                            reid_results_html = gr.HTML(
                                value='<div class="wid-empty-state"><strong>No results yet.</strong> Upload an image and press Identify.</div>',
                                label="Re-identification ranking",
                                elem_id="wid-reid-results",
                            )
                            output_text = gr.Textbox(
                                label="Detection report",
                                lines=12,
                                elem_id="wid-detection-report",
                                buttons=["copy"],
                            )

                    gr.HTML('<hr class="wid-divider">')
                    gr.Markdown("#### Top candidate reference images")
                    gr.HTML(
                        '<p class="wid-threshold-note">'
                        'Reference images for the highest-ranked candidates from the database. '
                        'Displayed only when a verified tiger crop is found.'
                        '</p>'
                    )
                    reference_gallery = gr.Gallery(
                        label="Reference images of top candidates",
                        columns=5,
                        height="auto",
                        elem_id="wid-ref-gallery",
                        elem_classes=["wid-gallery"],
                    )

                    gr.HTML('<hr class="wid-divider">')
                    gr.Markdown("#### Submit image to pending queue")
                    gr.HTML(
                        '<p class="wid-threshold-note">'
                        'Submit a newly observed image to the pending queue. '
                        'Requires reviewer confirmation before it affects matching.'
                        '</p>'
                    )
                    with gr.Row():
                        tiger_name_input = gr.Textbox(
                            label="Tiger name / ID",
                            placeholder="e.g. tiger_275",
                            scale=2,
                            elem_id="wid-submit-name",
                        )
                        provenance_input = gr.Textbox(
                            label="Provenance / notes",
                            placeholder="Camera trap ID, location, timestamp",
                            scale=2,
                            elem_id="wid-submit-provenance",
                        )
                        add_btn = gr.Button(
                            "Submit to pending queue",
                            variant="secondary",
                            scale=1,
                            elem_id="wid-submit-btn",
                        )
                    add_status = gr.Textbox(
                        label="",
                        lines=4,
                        show_label=False,
                        elem_id="wid-submit-status",
                    )

                    identify_btn.click(
                        fn=identify_tiger,
                        inputs=input_image,
                        outputs=[output_text, detection_annotated_img, reid_results_html, reference_gallery],
                        concurrency_limit=1,
                        api_name="identify_tiger",
                    )

                    add_btn.click(
                        fn=add_new_tiger,
                        inputs=[tiger_name_input, input_image, provenance_input],
                        outputs=add_status,
                        concurrency_limit=1,
                    )

                # ── 2: Batch ─────────────────────────────────────────────────────
                with gr.Tab("Batch", elem_id="wid-tab-batch"):
                    gr.Markdown("#### Batch candidate ranking")
                    gr.HTML(
                        '<p class="wid-threshold-note">'
                        f'Ranks all test images in data/test/ against the reference database using {_esc(SIMILARITY_LABEL.lower())}.'
                        '</p>'
                    )

                    batch_btn = gr.Button(
                        "Run batch ranking",
                        variant="primary",
                        elem_id="wid-batch-btn",
                    )
                    batch_output_html = gr.HTML(
                        value='<div class="wid-empty-state"><strong>No batch results.</strong> Press "Run batch ranking" to process all images in data/test/.</div>',
                        label="Batch results",
                        elem_id="wid-batch-output",
                    )
                    batch_download = gr.File(
                        label="Download results as CSV",
                        elem_id="wid-batch-download",
                        visible=True,
                    )

                    batch_btn.click(
                        fn=batch_identify,
                        outputs=[batch_output_html, batch_download],
                        concurrency_limit=1,
                    )

                # ── 3: Reference Gallery (placeholder) ───────────────────────────
                with gr.Tab("Reference Gallery", elem_id="wid-tab-refgallery"):
                    gr.HTML(
                        '<div class="wid-placeholder-card">'
                        '  <span class="wid-placeholder-badge">[ Under Construction ]</span>'
                        '  <h2 class="wid-placeholder-title">Reference Gallery</h2>'
                        '  <p class="wid-placeholder-desc">'
                        '    Will display all individuals from the loaded reference database as a '
                        '    browsable image grid. Each tile will show the tiger identity label, '
                        '    the number of reference images, and a thumbnail.'
                        '    Load the database on the <strong>Database</strong> page first.'
                        '  </p>'
                        '  <div class="wid-placeholder-grid-preview">'
                        '    <div class="wid-placeholder-item-box"><span>tiger_001</span><small>3 images</small></div>'
                        '    <div class="wid-placeholder-item-box"><span>tiger_002</span><small>2 images</small></div>'
                        '    <div class="wid-placeholder-item-box"><span>tiger_003</span><small>5 images</small></div>'
                        '    <div class="wid-placeholder-item-box"><span>&hellip;</span><small>placeholder</small></div>'
                        '  </div>'
                        '  <p class="wid-placeholder-note">'
                        '    This page is not yet functional. No data is shown here.'
                        '  </p>'
                        '</div>'
                    )

                # ── 4: Tadoba Species ─────────────────────────────────────────────
                with gr.Tab("Tadoba Species", elem_id="wid-tab-species"):
                    gr.Markdown("#### Verified Tadoba-Andhari species and model capabilities")
                    gr.Markdown(
                        "Lists verified species from ZSI Conservation Area Series (Vol 25), NTCA brief, "
                        "BirdLife/KBA factsheet 18278, and Maharashtra Forest Department checklist. "
                        "Model naming capability is evaluated directly from the model's verified label set."
                    )
                    refresh_species_btn = gr.Button(
                        "Inspect Tadoba species and capabilities",
                        variant="secondary",
                        elem_id="wid-species-btn",
                    )
                    species_output = gr.Textbox(
                        label="Verified species and model capabilities",
                        lines=24,
                        elem_id="wid-species-output",
                        buttons=["copy"],
                    )
                    refresh_species_btn.click(
                        fn=get_tadoba_species_table,
                        outputs=species_output,
                        concurrency_limit=1,
                    )

                # ── 5: Review Queue ──────────────────────────────────────────────
                with gr.Tab("Review Queue", elem_id="wid-tab-review"):
                    gr.Markdown("#### Pending submissions review queue")
                    gr.HTML(
                        '<p class="wid-threshold-note">'
                        'Submissions do not affect matching until confirmed by a reviewer.'
                        '</p>'
                    )

                    refresh_pending_btn = gr.Button(
                        "Refresh pending list",
                        variant="secondary",
                        elem_id="wid-refresh-pending-btn",
                    )
                    pending_list_output = gr.Textbox(
                        label="Pending submissions",
                        lines=8,
                        elem_id="wid-pending-output",
                    )
                    refresh_pending_btn.click(
                        fn=get_pending_summary,
                        outputs=pending_list_output,
                        concurrency_limit=1,
                    )

                    gr.HTML('<hr class="wid-divider">')
                    with gr.Row():
                        review_sub_id = gr.Textbox(
                            label="Submission ID",
                            placeholder="sub_...",
                            scale=2,
                            elem_id="wid-review-sub-id",
                        )
                        reviewer_note = gr.Textbox(
                            label="Reviewer note",
                            placeholder="Reason / notes",
                            scale=2,
                            elem_id="wid-reviewer-note",
                        )
                    with gr.Row():
                        approve_btn = gr.Button(
                            "Approve (add to reference database)",
                            variant="primary",
                            elem_id="wid-approve-btn",
                        )
                        reject_btn = gr.Button(
                            "Reject (move to trash)",
                            variant="secondary",
                            elem_id="wid-reject-btn",
                        )

                    review_status = gr.Textbox(
                        label="Review outcome",
                        lines=4,
                        elem_id="wid-review-status",
                    )

                    approve_btn.click(
                        fn=lambda sid, note: handle_review_submission(sid, "approve", note),
                        inputs=[review_sub_id, reviewer_note],
                        outputs=review_status,
                        concurrency_limit=1,
                    )
                    reject_btn.click(
                        fn=lambda sid, note: handle_review_submission(sid, "reject", note),
                        inputs=[review_sub_id, reviewer_note],
                        outputs=review_status,
                        concurrency_limit=1,
                    )

                # ── 6: Evaluation (placeholder) ──────────────────────────────────
                with gr.Tab("Evaluation", elem_id="wid-tab-evaluation"):
                    gr.HTML(
                        '<div class="wid-honesty-banner">'
                        '  <strong>Standing rule:</strong> No accuracy claims are shown here. '
                        '  Metrics will only appear after empirical evaluation on a documented '
                        '  dataset with published sample sizes and method. Cosine similarity '
                        '  is a ranking signal, not a probability or verified identity.'
                        '</div>'
                        '<div class="wid-placeholder-card">'
                        '  <span class="wid-placeholder-badge">[ Under Construction ]</span>'
                        '  <h2 class="wid-placeholder-title">Evaluation</h2>'
                        '  <p class="wid-placeholder-desc">'
                        '    Will display empirically measured re-identification metrics '
                        '    (Rank-1, mAP) once evaluated on a documented dataset. '
                        '    No values are displayed here until that evaluation is complete '
                        '    and the method and sample size are on record.'
                        '  </p>'
                        '  <div class="wid-placeholder-metrics-preview">'
                        '    <div class="wid-placeholder-metric">'
                        '      <strong>&mdash;</strong><span>Rank-1</span><small>not yet measured</small>'
                        '    </div>'
                        '    <div class="wid-placeholder-metric">'
                        '      <strong>&mdash;</strong><span>mAP</span><small>not yet measured</small>'
                        '    </div>'
                        '    <div class="wid-placeholder-metric">'
                        '      <strong>&mdash;</strong><span>Dataset</span><small>not yet documented</small>'
                        '    </div>'
                        '  </div>'
                        '  <p class="wid-placeholder-note">'
                        '    This page is not yet functional. No data is shown here.'
                        '  </p>'
                        '</div>'
                    )

                # ── 7: Database (Setup + Delete) ─────────────────────────────────
                with gr.Tab("Database", elem_id="wid-tab-setup"):
                    gr.Markdown("#### Database management")
                    gr.HTML(
                        '<p class="wid-threshold-note">'
                        'Load the existing .npz database, or rebuild from reference images in data/reference/.'
                        '</p>'
                    )

                    with gr.Row():
                        load_btn = gr.Button(
                            "Load existing database",
                            variant="primary",
                            elem_id="wid-load-btn",
                        )
                        build_btn = gr.Button(
                            "Build database from images",
                            variant="secondary",
                            elem_id="wid-build-btn",
                        )

                    setup_output = gr.Textbox(
                        label="STATUS",
                        lines=14,
                        elem_id="wid-setup-output",
                        buttons=["copy"],
                    )

                    load_btn.click(
                        fn=load_existing_database,
                        outputs=[setup_output, header_html],
                        concurrency_limit=1,
                    )
                    build_btn.click(
                        fn=build_reference_database,
                        outputs=[setup_output, header_html],
                        concurrency_limit=1,
                    )

                    # ── Destructive zone ──────────────────────────────────────────
                    gr.HTML('<hr class="wid-divider">')
                    gr.HTML(
                        '<div class="wid-destructive-zone">'
                        '<p class="wid-destructive-label">Destructive action</p>'
                        '<p class="wid-threshold-note" style="margin-top:0;">'
                        'Moves an individual folder to trash/. The database is updated immediately. '
                        'This cannot be undone from the UI.'
                        '</p>'
                    )
                    with gr.Row():
                        delete_name_input = gr.Textbox(
                            label="Tiger name / ID to delete",
                            placeholder="e.g.  tiger_030",
                            scale=2,
                            elem_id="wid-delete-name",
                        )
                        confirm_delete_box = gr.Checkbox(
                            label="Confirm moving tiger to trash",
                            value=False,
                            scale=1,
                            elem_id="wid-delete-confirm",
                        )
                        delete_btn = gr.Button(
                            "Delete tiger",
                            variant="secondary",
                            scale=1,
                            elem_id="wid-delete-btn",
                        )
                    delete_output = gr.Textbox(
                        label="",
                        lines=6,
                        show_label=False,
                        elem_id="wid-delete-output",
                    )
                    gr.HTML('</div>')  # close destructive zone

                    delete_btn.click(
                        fn=delete_tiger,
                        inputs=[delete_name_input, confirm_delete_box],
                        outputs=delete_output,
                        concurrency_limit=1,
                    )

                # ── 8: About ─────────────────────────────────────────────────────
                with gr.Tab("About", elem_id="wid-tab-info"):
                    gr.Markdown(f"""
    #### Method

    This system couples animal localization and species verification with individual tiger
    re-identification. A camera-trap image passes through four stages before any identity
    ranking is produced.

    #### Honesty guidelines

    - **{SIMILARITY_LABEL}** is a ranking signal, not a probability, confidence, or verified individual identity.
    - {SIMILARITY_NOTE}
    - Threshold is currently set to `{SIMILARITY_THRESHOLD:.2f}` ({THRESHOLD_LABEL}).
    - Scores produced by localizers and classifiers are ranking heuristics, not calibrated probabilities.

    #### Architecture

    | Stage | Model | Notes |
    |---|---|---|
    | Localizer | SSDLite320-MobileNetV3 | BSD 3-Clause, COCO |
    | Privacy rule | (built-in) | People detected and suppressed; never cropped, stored, or re-identified |
    | Species classifier | MobileNetV3-Small | ImageNet-1k |
    | Re-ID embedding | ViT-Base (vit_base_patch16_224) | 128-dimensional embedding |

    #### Database

    - Storage: atomic `.npz` vector matrix + companion JSON metadata.
    - Concurrency: Gradio queue (max_size={QUEUE_MAX_SIZE}, concurrency={DEFAULT_CONCURRENCY_LIMIT}), max upload {MAX_UPLOAD_SIZE_MB} MB, {MAX_PIXEL_COUNT:,} px limit.

    #### Limitations

    - Re-identification operates on cosine similarity ranking only. No accuracy claims are made without empirical evaluation on a documented dataset.
    - The threshold ({SIMILARITY_THRESHOLD:.2f}) is a configurable heuristic, not validated.
    - Results require expert review before any conservation decision.

    #### Licence and attribution

    See THIRD_PARTY_NOTICES.md in the project root for model licences and data attribution.
                    """)

        # ── Close main layout ─────────────────────────────────────────────────────
    # wid-shell-bottom removed: gr.Row closes automatically via Python context manager

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="wid-footer">'
        '<p>WILD-ID &mdash; Individual tiger re-identification &nbsp;|&nbsp; '
        '<a href="https://github.com/Adityathere06/Wild-ID" '
        'target="_blank" rel="noopener noreferrer">Repository</a></p>'
        '</div>'
    )



# Configure queue with bounded size and default concurrency
demo.queue(
    max_size=QUEUE_MAX_SIZE,
    default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wild-ID Gradio Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    initialize_system()
    load_existing_database()

    logger.info(
        f"Launching Wild-ID on {args.host}:{args.port} "
        f"(queue max_size={QUEUE_MAX_SIZE}, concurrency={DEFAULT_CONCURRENCY_LIMIT}, "
        f"max_file_size={MAX_UPLOAD_SIZE_MB}mb, timeout={REQUEST_TIMEOUT_SECONDS}s)"
    )

    js_code = Path(JS_PATH).read_text(encoding="utf-8") if Path(JS_PATH).exists() else None

    demo.launch(
        share=False,
        server_name=args.host,
        server_port=args.port,
        max_file_size=f"{MAX_UPLOAD_SIZE_MB}mb",
        show_error=True,
        favicon_path=FAVICON_PATH,
        css_paths=[CSS_PATH],
        js=js_code,
        allowed_paths=[str(BASE_DIR / "static")],
        footer_links=[],
    )
