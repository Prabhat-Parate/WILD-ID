"""Configuration settings for tiger re-identification.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = str(BASE_DIR / "data" / "reference")
PENDING_DIR = str(BASE_DIR / "data" / "pending")
TRASH_DIR = str(BASE_DIR / "trash")
TEST_DIR = str(BASE_DIR / "data" / "test")

# Model & Database paths
MODELS_DIR = str(BASE_DIR / "models")
WEIGHTS_PATH = str(BASE_DIR / "models" / "tiger_finetuned_vit.pth")
NPZ_PATH = str(BASE_DIR / "models" / "reference_features.npz")
METADATA_JSON_PATH = str(BASE_DIR / "models" / "reference_features.json")
LEGACY_PKL_PATH = str(BASE_DIR / "models" / "reference_features.pkl")

# Threshold configuration
# NOTE: Configurable heuristic, not validated until Task 2 produces empirical numbers.
SIMILARITY_THRESHOLD = 0.96
THRESHOLD_LABEL = "configurable heuristic, not validated"

# Top candidate retrieval count
TOP_K_CANDIDATES = 5

# Terminology & Honesty disclaimers
SIMILARITY_LABEL = "Cosine similarity"
SIMILARITY_NOTE = "Similarity indicates visual resemblance, not verified individual identity."
OUTCOME_ABOVE_THRESHOLD = "Candidate above threshold (unverified)"
OUTCOME_BELOW_THRESHOLD = "No candidate above threshold"

# Identity Name Validation
NAME_REGEX = r"^[a-zA-Z0-9_-]+$"
NAME_MIN_LENGTH = 1
NAME_MAX_LENGTH = 64

# ── Production Hardening ──────────────────────────────────────────────────────

# Server binding
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7861

# Image safety limits
MAX_UPLOAD_SIZE_MB = 20           # Reject uploads larger than this
MAX_PIXEL_COUNT = 30_000_000      # ~30 megapixels; reject decompression bombs
MAX_DISPLAY_PIXELS = 2_048_000    # Downscale to roughly 1600x1280 for processing
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Gradio queue & concurrency
QUEUE_MAX_SIZE = 8                # Reject fast when queue is full
DEFAULT_CONCURRENCY_LIMIT = 1     # One inference at a time (CPU bound)

# Torch threading
TORCH_NUM_THREADS = 4             # torch.set_num_threads value (CPU only)

# Logging
LOG_DIR = str(BASE_DIR / "logs")
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT = 3              # Keep 3 rotated backups

# Per-request timeout (seconds) for inference functions
REQUEST_TIMEOUT_SECONDS = 60

# Known file hashes for integrity checks (SHA-256)
EXPECTED_HASHES = {
    "models/tiger_finetuned_vit.pth": "4fef4c5552c027d25543179c2ae59afd438a40ce73ca996b2d6b53618f99f6fe",
    "models/reference_features.npz": "bd0bcb22b579e394b0733299cca62e9a98f2dae30208f77f8ecf9c32bc1e5a92",
    "models/reference_features.json": "63080f889bbd4b5d8350a5b7fd4fde9c86fbf633d3a96c863592a9188129d5d9",
    "data/tadoba-species.json": "f3df1746c6966e456da192328e533b5483b01a784c95ee5ffb4da8919c898820",
}
