import gradio as gr
import os
import sys
import time
import json
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
    LEGACY_PKL_PATH
)
from src.feature_extractor import TigerFeatureExtractor
from src.matcher import TigerMatcher
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
    move_to_trash,
    submit_to_pending,
    list_pending_submissions,
    review_pending_submission
)

# Global runtime state
extractor = None
matcher = None
detector = None
classifier = None
reference_features = {}


def initialize_system():
    global extractor, matcher, detector, classifier
    if extractor is None:
        extractor = TigerFeatureExtractor(device='cpu')
    if matcher is None:
        matcher = TigerMatcher()
    if detector is None:
        detector = AnimalDetector(device='cpu')
    if classifier is None:
        classifier = SpeciesClassifier(device='cpu')


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
        )

    reference_features = load_features()

    if not reference_features:
        return "Database file exists but is empty. Rebuild it."

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
    return result


def build_reference_database():
    """Scan data/reference/, extract embeddings, atomically save NPZ + JSON."""
    global extractor, matcher, reference_features

    if extractor is None:
        initialize_system()

    matcher = TigerMatcher()
    reference_features = {}

    individuals = parse_reference_structure(REFERENCE_DIR)
    if not individuals:
        yield "No reference images found. Add images to data/reference/"
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
            f"BUILDING DATABASE...\n"
            f"{'─'*40}\n\n"
            f"Processing : {os.path.basename(img_path)}\n"
            f"Progress   : {done}/{total} images done\n\n"
            f"Elapsed    : {fmt_time(elapsed)}\n"
            f"Remaining  : {remaining_str}"
        )
        yield live_status

        features = extractor.extract_features(img_path)
        if features is not None:
            matcher.add_reference(name, features, img_path)
            reference_features[img_path] = features

    save_features_npz(reference_features, NPZ_PATH, METADATA_JSON_PATH)

    total_elapsed = time.time() - build_start
    stats = matcher.get_statistics()

    result = (
        f"Database built and saved atomically.\n"
        f"{'─'*40}\n\n"
        f"Individuals loaded : {stats['total_individuals']}\n"
        f"Total images       : {stats['total_images']}\n"
        f"Total time taken   : {fmt_time(total_elapsed)}\n\n"
        f"Known tigers ({stats['total_individuals']}):\n" +
        "\n".join(f"  {name}" for name in sorted(stats['individuals']))
    )
    yield result


def delete_tiger(tiger_name, confirm_delete=False):
    """Move a tiger to trash/ after explicit confirmation."""
    global matcher, reference_features

    tiger_name = (tiger_name or "").strip()
    if not tiger_name:
        return "Please enter a tiger name to delete."

    if not confirm_delete:
        return (
            "DELETION ABORTED: Explicit confirmation required.\n"
            "Please check the 'Confirm moving tiger to trash' box before proceeding."
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
    and gated individual re-identification.
    """
    global extractor, matcher, detector, classifier

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first.", None, []

    if image is None:
        return "Please upload an image to identify.", None, []

    try:
        if detector is None or classifier is None:
            initialize_system()

        rgb_image = ensure_rgb_image(image)

        # Execute frame pipeline
        frame_res = process_frame(
            image=rgb_image,
            detector=detector,
            classifier=classifier,
            reid_extractor=extractor,
            reid_matcher=matcher,
            score_threshold=0.25,
            max_detections=10
        )

        annotated_img = frame_res['annotated_image']
        detections = frame_res['detections']
        summary_text = frame_res['summary_text']
        tiger_crops = frame_res['tiger_crops']

        gallery_items = []

        # If any verified tiger was found, run full top-5 candidate retrieval
        tiger_detections = [d for d in detections if d.get('is_tiger')]
        if tiger_detections and tiger_crops:
            # Evaluate the primary tiger crop
            primary_crop = tiger_crops[0]
            features = extractor.extract_features(primary_crop)

            if features is not None:
                best_match, best_score, all_scores = matcher.find_match(features, threshold=SIMILARITY_THRESHOLD)
                top_candidates = matcher.get_top_candidates(features, top_k=TOP_K_CANDIDATES)

                outcome_header = (
                    f"OUTCOME: {OUTCOME_ABOVE_THRESHOLD}"
                    if best_match is not None
                    else f"OUTCOME: {OUTCOME_BELOW_THRESHOLD}"
                )

                reid_lines = [
                    "\n" + "═" * 55,
                    "INDIVIDUAL RE-IDENTIFICATION RANKING (TIGER CROP)",
                    "═" * 55,
                    outcome_header,
                    f"Threshold                  : {SIMILARITY_THRESHOLD:.4f} ({THRESHOLD_LABEL})",
                    f"Highest {SIMILARITY_LABEL.lower()}: {best_score:.4f}",
                ]
                if best_match is not None:
                    reid_lines.append(f"Top Candidate              : {best_match}")

                reid_lines.append(f"\nNote: {SIMILARITY_NOTE}\n")
                reid_lines.append(f"Top {len(top_candidates)} Candidates ranked by {SIMILARITY_LABEL.lower()}:")

                for i, cand in enumerate(top_candidates, 1):
                    name = cand["name"]
                    score = cand["score"]
                    bar = '█' * max(0, min(20, int(score * 20)))
                    reid_lines.append(f"  {i}. {name:15s}  {SIMILARITY_LABEL}: {score:.4f}  {bar}")

                    for ref_img in cand["reference_images"][:2]:
                        if os.path.exists(ref_img):
                            gallery_items.append((ref_img, f"#{i} {name} (Cosine sim: {score:.4f})"))

                summary_text += "\n" + "\n".join(reid_lines)

        return summary_text, annotated_img, gallery_items

    except Exception as e:
        return f"Error during processing: {str(e)}", None, []


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
        lines.append(f"  • ID: {sub_id} | Name: {t_name} | Date: {t_date}")
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
        "═" * 70,
        "Verified against: ZSI Conservation Area Series (Vol 25), NTCA brief,",
        "BirdLife/KBA factsheet 18278, and Maharashtra Forest Dept checklist.",
        "Scores and capabilities computed directly from loaded model label set.",
        "═" * 70,
        f"{'Common Name':28s} {'Scientific Name':26s} {'Model Can Name?':16s} {'Matched Label':18s}",
        "─" * 90
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
    global extractor, matcher

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first."

    test_images = load_images_from_directory(test_dir)
    if not test_images:
        return f"No test images found in {test_dir}"

    results = (
        f"BATCH RESULTS — {len(test_images)} images\n"
        f"{'─'*50}\n"
        f"Threshold: {SIMILARITY_THRESHOLD:.4f} ({THRESHOLD_LABEL})\n"
        f"Note: {SIMILARITY_NOTE}\n\n"
    )

    for img_path in test_images:
        features = extractor.extract_features(img_path)
        if features is not None:
            best_match, sim_score, _ = matcher.find_match(features, threshold=SIMILARITY_THRESHOLD)
            if best_match is not None:
                status = f"{OUTCOME_ABOVE_THRESHOLD}: {best_match}"
            else:
                top_cands = matcher.get_top_matches(features, top_k=1)
                top_cand_name = top_cands[0][0] if top_cands else "None"
                status = f"{OUTCOME_BELOW_THRESHOLD} (closest: {top_cand_name})"

            results += f"{os.path.basename(img_path)}\n"
            results += f"  → {status}  ({SIMILARITY_LABEL}: {sim_score:.4f})\n\n"

    return results


# ── CSS (No external font imports; uses native system fonts) ───────────────────
css = """
footer { display: none !important; }
.built-with { display: none !important; }
#svelte-announcer { display: none !important; }
.gradio-container > .footer { display: none !important; }

body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background: #0e0e0e !important;
    color: #d4d4d4 !important;
}

.header-block {
    border-bottom: 1px solid #2a2a2a;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1rem;
}

.header-block h1 {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    color: #e8e8e8 !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

.tab-nav button {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    transition: color 0.2s, border-color 0.2s !important;
}

.tab-nav button.selected {
    color: #d4d4d4 !important;
    border-bottom: 2px solid #d4d4d4 !important;
}

.tab-nav button:hover { color: #999 !important; }

button.primary {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.4rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    cursor: pointer !important;
}

button.primary:hover {
    background: #2a2a2a !important;
    border-color: #555 !important;
}

button.secondary {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: #888 !important;
    border: 1px solid #252525 !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}

button.secondary:hover {
    color: #aaa !important;
    border-color: #3a3a3a !important;
}

textarea, .gr-textbox textarea {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #b0b0b0 !important;
    line-height: 1.7 !important;
}

input[type="text"], .gr-textbox input {
    font-family: Consolas, "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #b0b0b0 !important;
}

.threshold-note {
    font-family: Consolas, "Courier New", Courier, monospace;
    font-size: 0.72rem;
    color: #666;
    letter-spacing: 0.05em;
    margin: 0.3rem 0 1rem 0;
}

.disclaimer-banner {
    font-family: Consolas, "Courier New", Courier, monospace;
    font-size: 0.72rem;
    background: #181818;
    border-left: 3px solid #666;
    padding: 0.5rem 0.8rem;
    color: #aaa;
    margin-bottom: 0.8rem;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1rem 0;
}
"""

# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Tiger Identification & Detection") as demo:

    gr.HTML("""
    <div class="header-block">
        <h1>Tiger Detection & Re-Identification System</h1>
    </div>
    """)

    # ── Setup ─────────────────────────────────────────────────────────────────
    with gr.Tab("Setup"):
        gr.Markdown("#### Database Management")

        with gr.Row():
            load_btn  = gr.Button("Load Existing Database (.npz)", variant="primary")
            build_btn = gr.Button("Build Database from Images", variant="secondary")

        gr.HTML('<hr class="divider">')
        setup_output = gr.Textbox(label="Status", lines=14)

        load_btn.click(fn=load_existing_database, outputs=setup_output)
        build_btn.click(fn=build_reference_database, outputs=setup_output)

        gr.HTML('<hr class="divider">')
        gr.HTML('<p class="threshold-note">Delete a tiger — moves folder to trash/ after explicit confirmation</p>')
        with gr.Row():
            delete_name_input = gr.Textbox(
                label="Tiger Name / ID to Delete",
                placeholder="e.g.  tiger_030",
                scale=2
            )
            confirm_delete_box = gr.Checkbox(
                label="Confirm moving tiger to trash",
                value=False,
                scale=1
            )
            delete_btn = gr.Button("Delete Tiger", variant="secondary", scale=1)
        delete_output = gr.Textbox(label="", lines=6, show_label=False)

        delete_btn.click(
            fn=delete_tiger,
            inputs=[delete_name_input, confirm_delete_box],
            outputs=delete_output
        )

    # ── Identify & Detect ─────────────────────────────────────────────────────
    with gr.Tab("Detect & Identify"):
        gr.Markdown("#### Animal Localization, Privacy Suppression & Candidate Ranking")
        gr.HTML(
            f'<div class="disclaimer-banner">'
            f'<strong>Pipeline Stages:</strong> 1. Localizer (SSDLite320) &rarr; 2. Privacy Check (People suppressed) &rarr; 3. Species Verifier (MobileNetV3) &rarr; 4. Tiger Re-ID.<br>'
            f'<strong>Cosine similarity threshold:</strong> {SIMILARITY_THRESHOLD:.2f} ({THRESHOLD_LABEL}). Detection scores are ranking heuristics, not probabilities.<br>'
            f'<em>{SIMILARITY_NOTE}</em>'
            f'</div>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image  = gr.Image(label="Camera Trap Image", type="pil")
                identify_btn = gr.Button("Detect & Rank Candidates", variant="primary")
                detection_annotated_img = gr.Image(label="Detections (Boxes & Heuristic Scores)", type="pil")
            with gr.Column(scale=1):
                output_text  = gr.Textbox(label="Detection & Identification Report", lines=18)

        gr.HTML('<hr class="divider">')
        gr.Markdown("#### Top 5 Candidate Reference Images (Verified Tiger Only)")
        reference_gallery = gr.Gallery(label="Reference images of top candidates", columns=5, height="auto")

        gr.HTML('<hr class="divider">')
        gr.Markdown("#### Submit Image to Pending Queue")
        gr.HTML('<p class="threshold-note">Submit newly observed tiger image to pending queue. Requires reviewer confirmation before it affects matching.</p>')
        with gr.Row():
            tiger_name_input = gr.Textbox(
                label="Tiger Name / ID",
                placeholder="e.g. tiger_275",
                scale=2
            )
            provenance_input = gr.Textbox(
                label="Provenance / Notes",
                placeholder="Camera trap ID, location, timestamp",
                scale=2
            )
            add_btn = gr.Button("Submit to Pending Queue", variant="secondary", scale=1)
        add_status = gr.Textbox(label="", lines=4, show_label=False)

        identify_btn.click(
            fn=identify_tiger,
            inputs=input_image,
            outputs=[output_text, detection_annotated_img, reference_gallery]
        )

        add_btn.click(
            fn=add_new_tiger,
            inputs=[tiger_name_input, input_image, provenance_input],
            outputs=add_status
        )

    # ── Tadoba Species Reference ──────────────────────────────────────────────
    with gr.Tab("Tadoba Species"):
        gr.Markdown("#### Verified Tadoba-Andhari Species & Model Capabilities")
        gr.Markdown(
            "Lists verified species from ZSI Conservation Area Series (Vol 25), NTCA brief, "
            "BirdLife/KBA factsheet 18278, and Maharashtra Forest Department checklist. "
            "Model naming capability is evaluated directly from the model's verified label set."
        )
        refresh_species_btn = gr.Button("Inspect Tadoba Species & Capabilities", variant="secondary")
        species_output = gr.Textbox(label="Verified Species and Model Capabilities", lines=24)
        refresh_species_btn.click(fn=get_tadoba_species_table, outputs=species_output)

    # ── Pending Submissions Review ────────────────────────────────────────────
    with gr.Tab("Review Pending"):
        gr.Markdown("#### Pending Submissions Review Queue")
        gr.Markdown("Submissions do not affect matching until confirmed by a reviewer.")

        refresh_pending_btn = gr.Button("Refresh Pending List", variant="secondary")
        pending_list_output = gr.Textbox(label="Pending Submissions", lines=8)
        refresh_pending_btn.click(fn=get_pending_summary, outputs=pending_list_output)

        gr.HTML('<hr class="divider">')
        with gr.Row():
            review_sub_id = gr.Textbox(label="Submission ID", placeholder="sub_...", scale=2)
            reviewer_note = gr.Textbox(label="Reviewer Note", placeholder="Reason / notes", scale=2)
        with gr.Row():
            approve_btn = gr.Button("Approve (Add to Reference DB)", variant="primary")
            reject_btn  = gr.Button("Reject (Move to Trash)", variant="secondary")

        review_status = gr.Textbox(label="Review Outcome", lines=4)

        approve_btn.click(
            fn=lambda sid, note: handle_review_submission(sid, "approve", note),
            inputs=[review_sub_id, reviewer_note],
            outputs=review_status
        )
        reject_btn.click(
            fn=lambda sid, note: handle_review_submission(sid, "reject", note),
            inputs=[review_sub_id, reviewer_note],
            outputs=review_status
        )

    # ── Batch ─────────────────────────────────────────────────────────────────
    with gr.Tab("Batch"):
        gr.Markdown("#### Batch Candidate Ranking")
        gr.Markdown("Ranks all test images in `data/test/` against the reference database using cosine similarity.")
        batch_btn    = gr.Button("Run Batch Ranking", variant="primary")
        batch_output = gr.Textbox(label="Batch Results", lines=22)
        batch_btn.click(fn=batch_identify, outputs=batch_output)

    # ── Info ──────────────────────────────────────────────────────────────────
    with gr.Tab("Info"):
        gr.Markdown(f"""
#### About

This system couples an animal localization and species verification stage with individual tiger re-identification.

#### Important Honesty Guidelines

- **Cosine similarity** is a ranking signal, not a probability, confidence, or verified individual identity.
- {SIMILARITY_NOTE}
- Threshold is currently set to `{SIMILARITY_THRESHOLD:.2f}` ({THRESHOLD_LABEL}).
- Scores produced by localizers and classifiers are ranking heuristics, not calibrated probabilities.

#### Detection & Verification Architecture

- **Localizer** — SSDLite320-MobileNetV3 (BSD 3-Clause, COCO), ~140ms on CPU
- **Privacy Rule** — People are detected and suppressed immediately; never cropped, stored, or re-identified.
- **Species Classifier** — MobileNetV3-Small (ImageNet-1k), ~24ms on CPU. Re-ID runs strictly on verified tiger crops.
- **Re-ID Embedding** — ViT-Base (vit_base_patch16_224), 128-dimensional embedding
- **Database Storage** — Atomic `.npz` vector matrix + companion JSON metadata
        """)

if __name__ == "__main__":
    initialize_system()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861, css=css)
