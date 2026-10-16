"""One-time migration script: Converts reference_features.pkl to reference_features.npz + JSON metadata.
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import LEGACY_PKL_PATH, NPZ_PATH, METADATA_JSON_PATH
from src.utils import save_features_npz, load_features_npz

def migrate_pickle_to_npz(
    pkl_path: str = LEGACY_PKL_PATH,
    npz_path: str = NPZ_PATH,
    json_path: str = METADATA_JSON_PATH
) -> bool:
    print("[SECURITY WARNING] Loading pickle file. Pickle files can execute arbitrary code.")
    print("[SECURITY WARNING] Only load pickle files from verified and trusted sources.")

    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        print(f"[ERROR] Source pickle file not found: {pkl_path}")
        return False

    # Load legacy pickle
    with open(pkl_file, 'rb') as f:
        legacy_data = pickle.load(f)

    if not isinstance(legacy_data, dict):
        print(f"[ERROR] Expected dictionary in pickle file, got {type(legacy_data)}")
        return False

    print(f"Loaded {len(legacy_data)} feature entries from {pkl_path}")

    # Validate and standardize
    standardized_data = {}
    for key, val in legacy_data.items():
        arr = np.asarray(val, dtype=np.float32)
        standardized_data[str(key)] = arr

    # Save to atomic npz + JSON
    save_features_npz(standardized_data, npz_path=npz_path, json_path=json_path)

    # Verification read-back
    verified_data = load_features_npz(npz_path=npz_path, json_path=json_path)
    if len(verified_data) != len(standardized_data):
        print(f"[ERROR] Verification failed: count mismatch ({len(verified_data)} vs {len(standardized_data)})")
        return False

    # Check key and array equality
    for k in standardized_data:
        if k not in verified_data:
            print(f"[ERROR] Verification failed: missing key '{k}'")
            return False
        if not np.allclose(standardized_data[k], verified_data[k], atol=1e-6):
            print(f"[ERROR] Verification failed: vector mismatch for key '{k}'")
            return False

    print(f"[OK] Migration completed and verified successfully.")
    print(f"     NPZ file     : {npz_path}")
    print(f"     Metadata file: {json_path}")
    print(f"     Total entries: {len(verified_data)}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate reference features from pickle to NPZ + JSON.")
    parser.add_argument("--input", default=LEGACY_PKL_PATH, help="Path to input .pkl file")
    parser.add_argument("--output-npz", default=NPZ_PATH, help="Path to output .npz file")
    parser.add_argument("--output-json", default=METADATA_JSON_PATH, help="Path to output .json file")
    args = parser.parse_args()

    success = migrate_pickle_to_npz(args.input, args.output_npz, args.output_json)
    if not success:
        sys.exit(1)
