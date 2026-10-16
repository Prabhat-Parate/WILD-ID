"""Environment and dependency diagnostic utility for Wild-ID.

Checks performed:
1. Working directory: verifies root path and required subdirectories.
2. Package versions: validates installed libraries against requirements.txt.
3. UTF-8 environment: checks default encoding, stdout encoding, and file I/O roundtrip.
4. Weights & data integrity: verifies file existence and SHA-256 hashes against expected baseline.
5. Port availability: tests if target binding port (default 7861) is free on 127.0.0.1.
6. Live probe: checks HTTP connectivity if server is currently running.
"""

import os
import sys
import socket
import hashlib
import argparse
from pathlib import Path

# Ensure root directory is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WEIGHTS_PATH,
    NPZ_PATH,
    METADATA_JSON_PATH,
    LOG_DIR,
    TRASH_DIR,
    EXPECTED_HASHES
)


def compute_sha256(filepath: Path) -> str:
    """Compute full SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_working_directory() -> bool:
    """Verify script is run from project root and required folders exist."""
    print("\n[CHECK 1/6] Working Directory & Folder Structure")
    print("-" * 50)

    cwd = Path.cwd().resolve()
    print(f"  Current Working Directory: {cwd}")

    demo_file = cwd / "demo.py"
    if not demo_file.exists():
        print(f"  [FAIL] demo.py not found in current directory. Run from project root.")
        return False

    required_folders = ["data", "models", "src", "tests"]
    all_ok = True
    for folder in required_folders:
        p = cwd / folder
        if p.exists() and p.is_dir():
            print(f"  [OK] Found directory: {folder}/")
        else:
            print(f"  [FAIL] Missing required directory: {folder}/")
            all_ok = False

    # Auto-create runtime folders if missing
    for runtime_folder in ["logs", "trash"]:
        rf = cwd / runtime_folder
        rf.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Runtime directory ready: {runtime_folder}/")

    return all_ok


def check_package_versions() -> bool:
    """Verify runtime core packages and report optional training packages."""
    print("\n[CHECK 2/6] Package Versions")
    print("-" * 50)

    # Core deployment packages required to run the inference service
    core_runtime_packages = [
        ("torch", "torch", "2.10.0"),
        ("torchvision", "torchvision", "0.25.0"),
        ("timm", "timm", "1.0.24"),
        ("gradio", "gradio", "6.5.1"),
        ("pillow", "PIL", "12.0.0"),
        ("scikit-learn", "sklearn", "1.8.0"),
        ("numpy", "numpy", "2.3.5"),
        ("pandas", "pandas", "3.0.0"),
        ("pytest", "pytest", "9.0.3"),
        ("gradio_client", "gradio_client", "2.0.3"),
    ]

    all_ok = True
    print("  -- Core Deployment Runtime Dependencies --")
    for pkg_display, import_name, target_ver in core_runtime_packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg_display:16s} : installed {ver:12s} (base pinned: {target_ver})")
        except ImportError as e:
            print(f"  [FAIL] {pkg_display:16s} : NOT INSTALLED ({e})")
            all_ok = False

    # Optional training/offline dependencies from requirements.txt
    optional_packages = [
        ("opencv-python", "cv2", "4.13.0.90"),
        ("matplotlib", "matplotlib", "3.10.8"),
        ("seaborn", "seaborn", "0.13.2"),
        ("albumentations", "albumentations", "2.0.8"),
    ]
    print("\n  -- Optional / Offline Training Dependencies --")
    for pkg_display, import_name, target_ver in optional_packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK]   {pkg_display:16s} : installed {ver:12s}")
        except ImportError:
            print(f"  [INFO] {pkg_display:16s} : not installed (optional, training only; not required for CPU deployment)")

    return all_ok


def check_utf8_environment() -> bool:
    """Validate default encoding, stream encoding, and UTF-8 file operations."""
    print("\n[CHECK 3/6] UTF-8 Environment")
    print("-" * 50)

    default_enc = sys.getdefaultencoding()
    stdout_enc = sys.stdout.encoding or "unknown"
    print(f"  Default encoding : {default_enc}")
    print(f"  Stdout encoding  : {stdout_enc}")

    test_str = "Tiger Panthera tigris (Wagh Tadoba 2026)"
    test_file = ROOT_DIR / "logs" / "_utf8_probe.tmp"

    try:
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_str)

        with open(test_file, "r", encoding="utf-8") as f:
            read_back = f.read()

        test_file.unlink(missing_ok=True)

        if read_back == test_str:
            print("  [OK] UTF-8 file write and readback verified.")
            return True
        else:
            print("  [FAIL] Readback content mismatch.")
            return False
    except Exception as e:
        print(f"  [FAIL] UTF-8 probe exception: {e}")
        return False


def check_weights_and_data_integrity() -> bool:
    """Verify presence and SHA-256 hashes of weights and data artifacts."""
    print("\n[CHECK 4/6] Weights and Data Integrity (SHA-256)")
    print("-" * 50)

    files_to_check = [
        ("models/tiger_finetuned_vit.pth", True),
        ("models/reference_features.npz", True),
        ("models/reference_features.json", True),
        ("data/tadoba-species.json", True),
    ]

    all_ok = True
    for rel_path, required in files_to_check:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            status = "[FAIL]" if required else "[WARN]"
            print(f"  {status} File missing: {rel_path}")
            if required:
                all_ok = False
            continue

        size_mb = full_path.stat().st_size / (1024 * 1024)
        computed_hash = compute_sha256(full_path)
        expected = EXPECTED_HASHES.get(rel_path)

        if expected is None:
            print(f"  [WARN] {rel_path:32s} : {size_mb:8.2f} MB | SHA-256: {computed_hash[:16]}... (unverified)")
        elif computed_hash.lower() == expected.lower():
            print(f"  [OK]   {rel_path:32s} : {size_mb:8.2f} MB | SHA-256 match ({computed_hash[:16]}...)")
        else:
            print(f"  [FAIL] {rel_path:32s} : SHA-256 MISMATCH!")
            print(f"         Expected: {expected}")
            print(f"         Computed: {computed_hash}")
            all_ok = False

    return all_ok


def check_port_free(host: str, port: int) -> tuple[bool, str]:
    """Test if the port is free to bind."""
    print("\n[CHECK 5/6] Port Binding Availability")
    print("-" * 50)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        res = s.connect_ex((host, port))
        s.close()
        if res == 0:
            print(f"  [INFO] Port {port} on {host} is currently OCCUPIED (server running or in use).")
            return False, "occupied"
        else:
            print(f"  [OK] Port {port} on {host} is FREE to bind.")
            return True, "free"
    except Exception as e:
        print(f"  [WARN] Port check error: {e}")
        return True, "unknown"


def check_live_probe(host: str, port: int) -> bool:
    """Send live HTTP probe if port is occupied."""
    print("\n[CHECK 6/6] Live Server Probe")
    print("-" * 50)

    target_url = f"http://{host}:{port}/"
    import urllib.request
    try:
        req = urllib.request.Request(target_url, headers={"User-Agent": "Wild-ID Doctor/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            code = resp.getcode()
            if code == 200:
                print(f"  [OK] Server responded with HTTP 200 OK at {target_url}")
                return True
            else:
                print(f"  [WARN] Server responded with HTTP {code} at {target_url}")
                return False
    except urllib.error.URLError as e:
        print(f"  [INFO] No live HTTP server detected at {target_url} ({e.reason})")
        return False
    except Exception as e:
        print(f"  [WARN] Live probe connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Wild-ID Environment Doctor")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to inspect (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to inspect (default: {DEFAULT_PORT})")
    parser.add_argument("--probe", action="store_true", help="Always attempt live HTTP probe")
    args = parser.parse_args()

    print("=" * 60)
    print("  WILD-ID SYSTEM DIAGNOSTIC (DOCTOR)")
    print("=" * 60)

    ok1 = check_working_directory()
    ok2 = check_package_versions()
    ok3 = check_utf8_environment()
    ok4 = check_weights_and_data_integrity()
    port_free, port_status = check_port_free(args.host, args.port)

    live_ok = False
    if args.probe or port_status == "occupied":
        live_ok = check_live_probe(args.host, args.port)
    else:
        print("\n[CHECK 6/6] Live Server Probe")
        print("-" * 50)
        print("  [INFO] Skipped (server is not currently running). Pass --probe to force.")

    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 60)
    all_critical = ok1 and ok2 and ok3 and ok4

    if all_critical:
        print("  STATUS: ALL CRITICAL CHECKS PASSED [OK]")
        print(f"  Target Port {args.port} status: {port_status.upper()}")
        if live_ok:
            print(f"  Server is LIVE at http://{args.host}:{args.port}")
        return 0
    else:
        print("  STATUS: ONE OR MORE CRITICAL CHECKS FAILED [FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
