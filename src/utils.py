import os
import re
import json
import time
import math
import uuid
import shutil
import warnings
import concurrent.futures
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    REFERENCE_DIR,
    PENDING_DIR,
    TRASH_DIR,
    NPZ_PATH,
    METADATA_JSON_PATH,
    LEGACY_PKL_PATH,
    NAME_REGEX,
    NAME_MIN_LENGTH,
    NAME_MAX_LENGTH,
    MAX_UPLOAD_SIZE_MB,
    MAX_PIXEL_COUNT,
    MAX_DISPLAY_PIXELS,
    ALLOWED_IMAGE_EXTENSIONS,
    REQUEST_TIMEOUT_SECONDS
)

# Protect against decompression bomb attacks at the Pillow level
Image.MAX_IMAGE_PIXELS = MAX_PIXEL_COUNT

# Reserved Windows file names
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10))
}

def validate_identity_name(name: str, base_dir: str = REFERENCE_DIR) -> tuple[bool, str]:
    """
    Strictly validate an individual tiger name and verify path boundaries.

    Rules:
      - Letters, digits, underscores, and hyphens only.
      - Length between NAME_MIN_LENGTH and NAME_MAX_LENGTH.
      - Not a Windows reserved name.
      - Resolved directory path must strictly remain inside base_dir.

    Returns:
      (True, sanitized_name) or (False, error_description)
    """
    if not isinstance(name, str):
        return False, "Identity name must be a string."

    sanitized = name.strip()
    if not sanitized:
        return False, "Identity name cannot be empty."

    if len(sanitized) < NAME_MIN_LENGTH or len(sanitized) > NAME_MAX_LENGTH:
        return False, f"Name length must be between {NAME_MIN_LENGTH} and {NAME_MAX_LENGTH} characters."

    if not re.match(NAME_REGEX, sanitized):
        return False, "Identity name must contain only letters, digits, underscores, and hyphens (no spaces or slashes)."

    if sanitized.upper() in WINDOWS_RESERVED_NAMES:
        return False, f"'{sanitized}' is a reserved device name."

    # Verify path containment
    try:
        base_path = Path(base_dir).resolve()
        target_path = (base_path / sanitized).resolve()
        if not target_path.is_relative_to(base_path) or target_path == base_path:
            return False, "Resolved path escapes the reference directory."
    except Exception as e:
        return False, f"Path resolution failed: {e}"

    return True, sanitized


def safe_process_image(
    image_input,
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
    max_pixels: int = MAX_PIXEL_COUNT,
    max_display_pixels: int = MAX_DISPLAY_PIXELS
) -> tuple[Image.Image, dict]:
    """
    Validate upload bounds, protect against decompression bombs,
    downscale oversized images, and convert safely to RGB.

    Args:
        image_input: File path (str/Path) or PIL.Image.Image.
        max_size_mb: Maximum allowed file size in MB.
        max_pixels: Maximum allowed pixel count (bomb protection).
        max_display_pixels: Downscale images exceeding this pixel count.

    Returns:
        (sanitized_image: PIL.Image.Image, metadata: dict)

    Raises:
        ValueError: If size or pixel limit exceeded, or invalid format.
    """
    size_bytes = -1
    filename = ""

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_input}")

        filename = path.name
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image file extension: '{path.suffix}'. "
                f"Allowed extensions: {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}"
            )

        size_bytes = path.stat().st_size
        max_bytes = max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"Image file size ({size_bytes / (1024*1024):.2f} MB) exceeds "
                f"maximum upload limit of {max_size_mb} MB."
            )

        try:
            with Image.open(path) as img:
                total_pixels = img.width * img.height
                if total_pixels > max_pixels:
                    raise ValueError(
                        f"Image pixel count ({total_pixels}) exceeds maximum safety limit "
                        f"of {max_pixels} pixels (decompression bomb protection)."
                    )
                rgb_img = img.convert('RGB')
        except Image.DecompressionBombError as e:
            raise ValueError(f"Decompression bomb detected and rejected: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to decode image file: {e}")

    elif isinstance(image_input, Image.Image):
        total_pixels = image_input.width * image_input.height
        if total_pixels > max_pixels:
            raise ValueError(
                f"Image pixel count ({total_pixels}) exceeds maximum safety limit "
                f"of {max_pixels} pixels (decompression bomb protection)."
            )
        try:
            rgb_img = image_input.convert('RGB')
        except Image.DecompressionBombError as e:
            raise ValueError(f"Decompression bomb detected and rejected: {e}")
        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    orig_w, orig_h = rgb_img.size
    total_pixels = orig_w * orig_h
    downscaled = False

    if total_pixels > max_display_pixels:
        scale = math.sqrt(max_display_pixels / total_pixels)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        downscaled = True

    metadata = {
        "filename": filename,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "width": rgb_img.width,
        "height": rgb_img.height,
        "mode": rgb_img.mode,
        "total_pixels": rgb_img.width * rgb_img.height,
        "downscaled": downscaled,
        "size_bytes": size_bytes
    }
    return rgb_img, metadata


def ensure_rgb_image(image_input) -> Image.Image:
    """
    Convert an image input (PIL Image or file path) to an RGB PIL Image in memory.
    Enforces upload caps, decompression bomb protection, and downscaling.
    """
    img, _ = safe_process_image(image_input)
    return img


def run_with_timeout(func, args=(), kwargs=None, timeout: float = REQUEST_TIMEOUT_SECONDS):
    """
    Execute a callable with a strict timeout to prevent hung worker threads.

    Raises:
        TimeoutError: If execution exceeds timeout seconds.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout:.1f} seconds")


def save_features_npz(features_dict: dict, npz_path: str = NPZ_PATH, json_path: str = METADATA_JSON_PATH):
    """
    Atomically save feature embeddings to a .npz file and metadata to a JSON file.
    Uses temporary files and atomic rename (os.replace).
    """
    target_npz = Path(npz_path).resolve()
    target_json = Path(json_path).resolve()

    target_npz.parent.mkdir(parents=True, exist_ok=True)
    target_json.parent.mkdir(parents=True, exist_ok=True)

    unique_suffix = f"{os.getpid()}_{time.time_ns()}"
    tmp_npz = target_npz.with_suffix(f".tmp_{unique_suffix}.npz")
    tmp_json = target_json.with_suffix(f".tmp_{unique_suffix}.json")

    image_paths = list(features_dict.keys())
    if image_paths:
        embeddings_matrix = np.stack([features_dict[p] for p in image_paths], axis=0).astype(np.float32)
    else:
        embeddings_matrix = np.empty((0, 128), dtype=np.float32)

    # Metadata entries
    entries = []
    for idx, path_str in enumerate(image_paths):
        parts = Path(path_str).parts
        tiger_name = parts[-2] if len(parts) >= 2 else "unknown"
        entries.append({
            "index": idx,
            "image_path": path_str,
            "tiger_name": tiger_name
        })

    metadata = {
        "format_version": 1,
        "count": len(image_paths),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries
    }

    try:
        # Save npz with explicit string dtype for keys
        keys_array = np.array(image_paths, dtype=str)
        np.savez_compressed(
            tmp_npz,
            embeddings=embeddings_matrix,
            keys=keys_array
        )

        # Save JSON metadata (UTF-8)
        with open(tmp_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replacement
        os.replace(tmp_npz, target_npz)
        os.replace(tmp_json, target_json)
        print(f"[OK] Features saved atomically to {target_npz} and {target_json}")

    except Exception:
        # Cleanup temporary files on failure
        if tmp_npz.exists():
            tmp_npz.unlink(missing_ok=True)
        if tmp_json.exists():
            tmp_json.unlink(missing_ok=True)
        raise


def load_features_npz(npz_path: str = NPZ_PATH, json_path: str = METADATA_JSON_PATH) -> dict:
    """
    Load feature embeddings from .npz and JSON metadata without pickle.
    Returns {image_path: embedding_vector}.
    """
    npz_file = Path(npz_path)
    if not npz_file.exists():
        print(f"File {npz_path} does not exist")
        return {}

    with np.load(npz_file, allow_pickle=False) as data:
        embeddings = data['embeddings']

        json_file = Path(json_path) if json_path else npz_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            keys = [entry['image_path'] for entry in meta.get('entries', [])]
        else:
            keys = [str(k) for k in data['keys']]

        features_dict = {}
        for i, key in enumerate(keys):
            features_dict[key] = embeddings[i]

    print(f"[OK] Features loaded from {npz_path} ({len(features_dict)} entries)")
    return features_dict


def save_features(features_dict: dict, filepath: str = None):
    """
    Save features. Defaults to atomic .npz + .json format.
    """
    if filepath and filepath.endswith('.pkl'):
        # For compatibility with legacy calls, also write .npz and JSON
        npz_target = str(Path(filepath).with_suffix('.npz'))
        json_target = str(Path(filepath).with_suffix('.json'))
        save_features_npz(features_dict, npz_target, json_target)
    else:
        save_features_npz(features_dict, NPZ_PATH, METADATA_JSON_PATH)


def load_features(filepath: str = None) -> dict:
    """
    Load features. Prefers .npz if available. Warns if loading legacy .pkl.
    """
    # If no path specified, prefer NPZ
    if filepath is None or filepath == LEGACY_PKL_PATH:
        if Path(NPZ_PATH).exists():
            return load_features_npz(NPZ_PATH, METADATA_JSON_PATH)
        filepath = LEGACY_PKL_PATH

    path_obj = Path(filepath)

    # Check if corresponding .npz exists
    npz_alt = path_obj.with_suffix('.npz')
    if npz_alt.exists():
        json_alt = path_obj.with_suffix('.json')
        return load_features_npz(str(npz_alt), str(json_alt))

    if not path_obj.exists():
        print(f"File {filepath} does not exist")
        return {}

    if path_obj.suffix.lower() == '.pkl':
        warnings.warn(
            "Loading pickle files is potentially insecure. Ensure file is from a trusted source.",
            UserWarning
        )
        print("[SECURITY WARNING] Loading pickle file. Pickle files must only be read from trusted sources.")
        import pickle
        with open(filepath, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"[OK] Features loaded from legacy pickle: {filepath}")
        return features_dict

    if path_obj.suffix.lower() == '.npz':
        json_file = str(path_obj.with_suffix('.json'))
        return load_features_npz(str(path_obj), json_file)

    return {}


def move_to_trash(target_path: str, trash_dir: str = TRASH_DIR) -> str:
    """
    Move a file or directory to the trash folder instead of destructive deletion.
    Returns the destination path in trash.
    """
    target = Path(target_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target_path}")

    trash = Path(trash_dir).resolve()
    trash.mkdir(parents=True, exist_ok=True)

    timestamp_str = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    dest = trash / f"{target.name}_{timestamp_str}"

    shutil.move(str(target), str(dest))
    return str(dest)


def submit_to_pending(image_input, tiger_name: str, provenance: dict = None, pending_dir: str = PENDING_DIR) -> tuple[bool, str, dict]:
    """
    Submit a newly observed tiger image to the pending queue with provenance metadata.
    This does NOT add to the reference database and does NOT affect matching.

    Returns:
      (success: bool, message: str, submission_metadata: dict)
    """
    is_valid, validated_name = validate_identity_name(tiger_name)
    if not is_valid:
        return False, validated_name, {}

    try:
        rgb_image = ensure_rgb_image(image_input)
    except Exception as e:
        return False, f"Failed to process image: {e}", {}

    pending_path = Path(pending_dir).resolve()
    pending_path.mkdir(parents=True, exist_ok=True)

    sub_id = f"sub_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    img_filename = f"{sub_id}.jpg"
    json_filename = f"{sub_id}.json"

    dest_img_path = pending_path / img_filename
    dest_json_path = pending_path / json_filename

    # Save pending image as RGB JPEG
    rgb_image.save(dest_img_path, format="JPEG")

    meta = {
        "submission_id": sub_id,
        "tiger_name": validated_name,
        "image_path": str(dest_img_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": time.time(),
        "status": "pending",
        "provenance": provenance or {"source": "user_submission"}
    }

    # Save pending metadata atomically
    tmp_json = dest_json_path.with_suffix(".tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_json, dest_json_path)

    msg = (
        f"Image submitted to pending set for review.\n"
        f"Submission ID : {sub_id}\n"
        f"Tiger Name    : {validated_name}\n"
        f"Status        : Pending reviewer confirmation (will not affect matching)."
    )
    return True, msg, meta


def list_pending_submissions(pending_dir: str = PENDING_DIR, max_items: int = 50) -> list[dict]:
    """List submissions currently in pending status, bounded to max_items."""
    pending_path = Path(pending_dir).resolve()
    if not pending_path.exists():
        return []

    submissions = []
    for json_file in sorted(pending_path.glob("sub_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("status") == "pending":
                    submissions.append(data)
                    if len(submissions) >= max_items:
                        break
        except Exception:
            continue
    return submissions


def review_pending_submission(
    submission_id: str,
    action: str,
    reviewer_note: str = "",
    extractor = None,
    matcher = None,
    reference_features: dict = None,
    pending_dir: str = PENDING_DIR,
    reference_dir: str = REFERENCE_DIR,
    trash_dir: str = TRASH_DIR,
    npz_path: str = NPZ_PATH,
    json_path: str = METADATA_JSON_PATH
) -> tuple[bool, str, dict]:
    """
    Review a pending submission. Action must be 'approve' or 'reject'.
    If approved: moves image into data/reference/<tiger_name>/, updates matcher and feature store.
    If rejected: moves pending image into trash/.
    """
    if action not in ("approve", "reject"):
        return False, f"Invalid review action '{action}'. Must be 'approve' or 'reject'.", {}

    pending_path = Path(pending_dir).resolve()
    meta_path = pending_path / f"{submission_id}.json"
    if not meta_path.exists():
        return False, f"Submission ID '{submission_id}' not found.", {}

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    if meta.get("status") != "pending":
        return False, f"Submission '{submission_id}' has already been processed (status: {meta.get('status')}).", meta

    img_path = Path(meta["image_path"])
    tiger_name = meta["tiger_name"]

    if action == "approve":
        is_valid, sanitized_name = validate_identity_name(tiger_name, reference_dir)
        if not is_valid:
            return False, f"Cannot approve: invalid name ({sanitized_name})", meta

        tiger_folder = Path(reference_dir).resolve() / sanitized_name
        tiger_folder.mkdir(parents=True, exist_ok=True)

        existing = [f for f in tiger_folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        dest_filename = f"img_{len(existing) + 1:03d}.jpg"
        final_dest = tiger_folder / dest_filename

        shutil.move(str(img_path), str(final_dest))

        # Update metadata
        meta["status"] = "approved"
        meta["approved_at"] = datetime.now(timezone.utc).isoformat()
        meta["final_path"] = str(final_dest)
        meta["reviewer_note"] = reviewer_note

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

        # Update matcher and reference features if supplied
        if extractor is not None and matcher is not None and reference_features is not None:
            features = extractor.extract_features(str(final_dest))
            if features is not None:
                matcher.add_reference(sanitized_name, features, str(final_dest))
                reference_features[str(final_dest)] = features
                save_features_npz(reference_features, npz_path=npz_path, json_path=json_path)

        msg = f"Submission '{submission_id}' approved. Added to reference database as {sanitized_name} ({final_dest.name})."
        return True, msg, meta

    else:  # reject
        dest_trash = move_to_trash(str(img_path), trash_dir)
        meta["status"] = "rejected"
        meta["rejected_at"] = datetime.now(timezone.utc).isoformat()
        meta["trash_path"] = dest_trash
        meta["reviewer_note"] = reviewer_note

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

        msg = f"Submission '{submission_id}' rejected and moved to trash."
        return True, msg, meta


def load_images_from_directory(directory):
    """Load sorted image paths from a directory."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return image_paths

    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(directory, filename))

    return sorted(image_paths)


def parse_reference_structure(reference_dir):
    """Parse reference directory structure into a mapping of tiger_name -> list[image_paths]."""
    individuals = {}

    if not os.path.exists(reference_dir):
        return individuals

    for individual_name in sorted(os.listdir(reference_dir)):
        individual_path = os.path.join(reference_dir, individual_name)
        if os.path.isdir(individual_path):
            images = load_images_from_directory(individual_path)
            if images:
                individuals[individual_name] = images

    return individuals


def validate_dataset(reference_dir, test_dir):
    """Validate directory structures for reference and test sets."""
    issues = []

    if not os.path.exists(reference_dir):
        issues.append(f"Reference directory '{reference_dir}' does not exist")
    else:
        individuals = parse_reference_structure(reference_dir)
        if not individuals:
            issues.append("No individuals found in reference directory")

    if not os.path.exists(test_dir):
        issues.append(f"Test directory '{test_dir}' does not exist")

    if issues:
        for issue in issues:
            print(f"[ERROR] {issue}")
        return False
    return True