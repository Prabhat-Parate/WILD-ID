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
