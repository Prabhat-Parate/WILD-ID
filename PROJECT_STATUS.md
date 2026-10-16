# Project Status: Wild-ID Tiger Re-Identification

This document tracks implementation status across the repository, categorized into **Implemented**, **Partial**, and **Planned** capabilities.

---

## 1. Status Overview

| Capability Area | Status | Key Components |
| :--- | :--- | :--- |
| **Atomic Database Storage** | Implemented | Compressed `.npz` + UTF-8 JSON metadata with atomic replacement (`os.replace`) |
| **Path & Identity Safety** | Implemented | Strict identity regex, Windows device name checks, directory containment validation |
| **Pending Review Quarantine** | Implemented | Newly submitted images quarantined in `data/pending/` until explicit reviewer approval |
| **Safe Deletion** | Implemented | Moves to `trash/` with timestamping; recursive/destructive deletions eliminated |
| **Animal Detection & Localization** | Implemented | SSDLite320-MobileNetV3 (BSD-3-Clause, ~140ms on CPU, bounded `max_detections=10`) |
| **Human Privacy Enforcement** | Implemented | Immediate suppression of human detections; never cropped, saved, or re-identified |
| **Species Verification** | Implemented | MobileNetV3-Small (ImageNet-1k, ~24ms on CPU); re-ID strictly gated on verified tiger |
| **Faunal Checklist & Capabilities** | Implemented | Verified Tadoba-Andhari fauna dynamically queried against model label set |
| **Production Queue & Concurrency** | Implemented | Gradio queue (`max_size=8`, default concurrency=1, explicit per-event `concurrency_limit=1`) |
| **Thread Management** | Implemented | `torch.set_num_threads(4)` configured on startup; CPU-only execution |
| **Upload Safety & Decompression Bombs** | Implemented | 20MB file cap, 30M pixel bomb threshold rejection, downscaling to <= 2.05M pixels |
| **Timeout Protection** | Implemented | Per-request inference wrapped with 60-second execution timeout |
| **Privacy-Preserving Logging** | Implemented | Rotating file handler (5MB, 3 backups); logs metadata only, zero image pixels |
| **System Diagnostics (Doctor)** | Implemented | `scripts/doctor.py` validates directories, packages, UTF-8, hashes, ports, live probe |
| **PowerShell Automation** | Implemented | `scripts/start.ps1` and root `start.ps1` with configurable `-Port` parameter |
| **Resilience & Load Testing** | Implemented | `scripts/load_test.py` with concurrent clients, malformed probes, and Windows NT memory instrumentation |
| **Automated Test Coverage** | Implemented | 32 automated pytest tests covering detection, safety, database, and hardening |
| **Reference Database** | Partial | Holdout set contains 33 individuals; baseline database initialized |
| **Similarity Threshold Calibration** | Partial | Configurable heuristic (`0.96`), labeled explicitly as unvalidated heuristic |
| **Flank Matching Disambiguation** | Planned | Automated classification of left vs. right flank to prevent cross-side false rejection |
| **Flank Quality & Pose Estimation** | Planned | Automated rejection of excessively angled, blurred, or occluded tiger flanks |
| **Tadoba Empirical Validation** | Planned | Formal ground-truth evaluation on verified wild camera-trap dataset from TATR |

---

## 2. Implemented Capabilities

### A. Data Integrity & Storage Safety
- **No Pickles for New Data**: Features are stored in compressed `.npz` arrays (`allow_pickle=False`) with human-readable JSON metadata.
- **Atomic Persistence**: Database writes write to unique temporary files, sync to disk (`os.fsync`), and atomically replace target files (`os.replace`), preventing corruption on sudden interruption.
- **Sanitized Directory Operations**: Tiger identifiers are restricted to `[a-zA-Z0-9_-]`, checked against Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`), and verified to remain within designated directory boundaries.

### B. Detection, Species Verification & Privacy
- **Real Two-Stage Gated Pipeline**: Camera trap frames are first localized with SSDLite320-MobileNetV3 and verified with MobileNetV3-Small. Tiger re-identification is strictly gated to verified tiger crops.
- **Zero Privacy Leakage**: Any frame containing human beings suppresses human crops immediately. No person is ever cropped, stored, or processed for identity embeddings.
- **Bounded Output**: Output boxes and top candidate rankings are strictly bounded (`max_detections=10`, `top_k=5`) to prevent memory exhaustion on crowded frames.

### C. Server Hardening & Concurrency
- **Bounded Gradio Queue**: Configured with `demo.queue(max_size=8, default_concurrency_limit=1)` to enforce single-worker serialized inference on CPU and fast 503 rejection when saturated.
- **Shared Singleton Models**: Extractor, matcher, detector, and classifier are instantiated once on startup and shared across all incoming requests.
- **Inference Mode**: All forward passes use `torch.inference_mode()` to eliminate autograd graph overhead and memory leaks.
- **Configured Thread Pool**: `torch.set_num_threads(4)` explicitly set for deterministic CPU parallelism without thread thrashing.
- **Safety Preprocessing**: Pillow decompression bombs (> 30M pixels) and oversized uploads (> 20MB) are rejected before memory allocation. Large images are downscaled preserving aspect ratio.
- **Request Timeouts**: Inference functions are wrapped in 60-second timeouts to avoid hanging worker threads.
- **Rotated Auditing**: Logs write to `logs/wild_id.log` with 5MB rotation and 3 backups, containing metadata only (timestamps, dimensions, scores), never logging raw pixels or base64 data.

### D. Tooling & Testing
- **Preflight Doctor (`scripts/doctor.py`)**: Checks environment, pinned versions, UTF-8 file roundtrips, SHA-256 weight hashes, port binding, and live HTTP probes.
- **Startup Script (`scripts/start.ps1`, `start.ps1`)**: Single-command startup with configurable port (`-Port 7861`), UTF-8 environment setup, and doctor preflight integration.
- **Load Testing Harness (`scripts/load_test.py`)**: Uses `gradio_client` to dispatch concurrent bursts (12 threads), mixes valid and malformed payloads, measures latency percentiles, tests post-burst recovery, and instruments physical working set memory via Windows NT kernel APIs (`ctypes.windll.psapi.GetProcessMemoryInfo`).

---

## 3. Partial / Work-in-Progress

1. **Similarity Threshold Calibration (`SIMILARITY_THRESHOLD = 0.96`)**:
   - Status: Configurable heuristic in `src/config.py`.
   - Notes: Terminology has been rigorously audited to present cosine similarity as a ranking signal, not a probability or verified identity. However, the numerical threshold of 0.96 has not yet been calibrated via empirical ROC curves on a verified camera-trap benchmark.
2. **Reference Database (`33 Holdout Tigers`)**:
   - Status: Database initialized from holdout partition.
   - Notes: Holds 33 individual profiles with 128-dimensional embeddings. Requires broader diversity of illumination and seasonal coat variations.

---

## 4. Planned Capabilities (Required Before Tadoba Field Claims)

The following capabilities are mandatory before making any claims regarding operational deployment or identification accuracy in the Tadoba-Andhari Tiger Reserve:

1. **TATR-Specific Ground-Truth Dataset**:
   - Assembly of an official, documented camera-trap dataset verified by field biologists and park authorities.
2. **Flank Disambiguation (Left vs. Right Flank)**:
   - Tiger stripe patterns are asymmetric. An individual's left flank does not match its right flank. An automated flank classifier is required to avoid false negative associations.
3. **Flank Quality & Occlusion Filtering**:
   - Camera traps frequently capture partial flanks, nocturnal IR glare, and heavy vegetation occlusion. An automated quality assessment module is needed to score usability prior to embedding extraction.
4. **Empirical Operating Point Validation**:
   - Systematic determination of False Acceptance Rate (FAR) and False Rejection Rate (FRR) across documented sample sizes under field conditions.
