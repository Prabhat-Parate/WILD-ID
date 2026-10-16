# Wild-ID: Individual Re-Identification of Tigers

A hardened, privacy-preserving animal detection and individual tiger re-identification system for camera-trap imagery.

---

## 1. Pipeline Architecture

Fine-grained individual pattern matching directly on uncurated full-frame captures risks false associations and privacy violations. This system executes a four-stage pipeline:

```
Camera Trap Image
       │
       ▼
[Stage 1: Localizer] ── SSDLite320-MobileNetV3 (BSD-3-Clause, COCO)
       │
       ├─ If person detected ──> [Stage 2: Privacy Filter] ──> SUPPRESS (Never cropped, stored, or identified)
       │
       └─ If animal detected ──> [Stage 3: Species Verifier] ── MobileNetV3-Small (ImageNet-1k)
                                       │
                                       ├─ Non-tiger / unverified ──> Report species / BYPASS re-ID
                                       │
                                       └─ Verified Tiger (Panthera tigris)
                                               │
                                               ▼
                                      [Stage 4: Tiger Re-ID Head] ── ViT-Base (128-dim embedding)
                                               │
                                               ▼
                                      Cosine Similarity Ranking vs Reference DB
                                               │
                                               ▼
                                      Candidate Identity (unverified) / No candidate above threshold
```

---

## 2. Training Summary & Methodology

The re-identification head is trained using metric learning (not classification):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Tiger Identities** | 107 | Curated split of individual tigers |
| **Training Split (70%)** | 74 tigers | Used to fine-tune embedding projection via Triplet Loss |
| **Holdout / Reference Split (30%)** | 33 tigers | Unseen during fine-tuning; populated in reference database |
| **Random Seed** | 42 | Deterministic split |
| **Backbone Architecture** | ViT-Base (`vit_base_patch16_224`) | Pre-trained on ImageNet-21k, last 4 transformer blocks unfrozen |
| **Embedding Head** | Linear(768->256) -> ReLU -> Linear(256->128) -> L2 | Produces 128-dimensional unit vector |
| **Epochs Trained** | 19 (early stopped, patience=10) | Optimizer: Adam (lr=0.0001), CosineAnnealingLR |
| **Best Triplet Loss** | 0.0054 (epoch 9) | Optimization convergence metric on 74-tiger training split |
| **Similarity Threshold** | Configurable heuristic (`0.96` in `src/config.py`) | Configurable heuristic; not an empirical validation metric |

> **IMPORTANT ACCURACY & HONESTY DISCLAIMER:**  
> Best triplet loss (`0.0054`) measures optimization convergence on the 74-tiger training split. It does **not** represent field re-identification accuracy, precision, or operational performance. Cosine similarity is a **ranking signal**, not a probability, confidence, or verified biological identity. This system has **not** been deployed or validated in the Tadoba-Andhari Tiger Reserve.

---

## 3. Project Structure

```
Wild-ID/
├── data/
│   ├── fine_tuned_tigers/     # 74 training tigers (not used at inference)
│   ├── reference/             # 33 holdout tigers (reference database)
│   ├── test/                  # Sample query images
│   ├── pending/               # Staging queue for new submissions awaiting review
│   └── tadoba-species.json    # Verified Tadoba-Andhari faunal checklist
├── models/
│   ├── tiger_finetuned_vit.pth# Fine-tuned ViT weights (328.2 MB)
│   ├── reference_features.npz # Atomic embeddings matrix (allow_pickle=False)
│   └── reference_features.json# Reference identity metadata
├── src/
│   ├── config.py              # Central configuration, safety caps, and hashes
│   ├── detector.py            # Localizer, privacy suppression & species verifier
│   ├── feature_extractor.py   # ViT 128-dim embedding extractor
│   ├── matcher.py             # Cosine similarity candidate retrieval
│   ├── logger.py              # Rotating privacy logger (no image pixels recorded)
│   └── utils.py               # Safe image loading, timeouts, atomic persistence
├── scripts/
│   ├── doctor.py              # Environment, dependency & file integrity diagnostics
│   ├── load_test.py           # Concurrency, malformed uploads & memory harness
│   └── start.ps1              # Configurable PowerShell startup script
├── tests/
│   ├── test_detection.py      # Detection, privacy & gating unit tests
│   ├── test_wild_id.py        # Database migration, pending flow & path safety tests
│   └── test_hardening.py      # Decompression bombs, upload caps, timeouts & memory
├── demo.py                    # Hardened Gradio web interface
├── start.ps1                  # Root convenience startup script
├── requirements.txt           # Pinned dependencies
├── MODEL_CARD.md              # Detailed pipeline specifications and limitations
├── DATASET.md                 # Dataset partition structure and provenance notes
├── THIRD_PARTY_NOTICES.md     # Third-party licensing and AGPL analysis
└── PROJECT_STATUS.md          # Implemented, partial, and planned capabilities
```

---

## 4. Windows Setup & Installation

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.13 (CPU only)
- PowerShell 5.1+

### Step-by-Step Installation

```powershell
# 1. Clone repository
git clone https://github.com/Adityathere06/Wild-ID.git
cd Wild-ID

# 2. Create virtual environment with Python 3.13
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 5. Download Required Model Weights & Data

Before running, download the required model weights and dataset:

### 1. Model Weights (`tiger_finetuned_vit.pth`, 328.2 MB)
Download and place inside `models/`:
- **[Download tiger_finetuned_vit.pth](https://drive.google.com/file/d/1GZ8hb2vX3JBfK89WqR7KZZsOHSaAdXuc/view?usp=sharing)**
- Expected SHA-256: `4fef4c5552c027d25543179c2ae59afd438a40ce73ca996b2d6b53618f99f6fe`

### 2. Dataset (`data.zip`)
Download, extract, and ensure `data/` exists at the repository root:
- **[Download data.zip](https://drive.google.com/file/d/1eWLyGOAd_Lt4wO07_dwzRsTHuahO5Af3/view?usp=sharing)**

---

## 6. Preflight Diagnostics & Verification

Run the environment doctor to verify directories, package versions, UTF-8 compatibility, SHA-256 weight hashes, and port availability:

```powershell
python scripts/doctor.py --port 7861
```

To probe a currently running server:
```powershell
python scripts/doctor.py --port 7861 --probe
```

---

## 7. Running the Server

Start the hardened Gradio server using the PowerShell startup script with a configurable port:

```powershell
# Default port (7861) with pre-flight check
.\start.ps1

# Custom port
.\start.ps1 -Port 8080

# Direct python invocation
python demo.py --host 127.0.0.1 --port 7861
```

Once running, navigate to: `http://127.0.0.1:7861`

### Runtime Hardening Features
- **Queue & Concurrency**: Gradio queue with `max_size=8`, default concurrency `1`, and per-event `concurrency_limit=1`.
- **Upload Safety**: Max upload size capped at `20 MB`. Images exceeding `30,000,000 pixels` (~30 MP) are rejected as decompression bombs. Images exceeding `2,048,000 pixels` are automatically downscaled.
- **Timeouts**: Inference wrapped in 60-second timeouts to prevent hung worker threads.
- **Privacy Logging**: Rotating logs (`logs/wild_id.log`, 5 MB, 3 backups) record metadata only, never raw image pixels.

---

## 8. Resilience & Load Testing

Run the automated load testing harness using `gradio_client` to stress concurrency, test fast rejection of malformed uploads, measure latency percentiles, verify post-burst recovery, and instrument process memory via Windows NT kernel APIs:

```powershell
# Run against active server
python scripts/load_test.py --host 127.0.0.1 --port 7861 --concurrency 12 --requests 24 --output-json logs/load_test_results.json
```

---

## 9. Running Tests

Run the complete 32-test automated suite:

```powershell
pytest
```
