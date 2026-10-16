# Model Card: Tiger Detection & Individual Re-Identification Pipeline

## 1. Overview & Objectives

In automated camera-trap monitoring, matching fine-grained stripe pattern embeddings directly on uncurated full-frame captures risks false associations, background noise distortion, and severe privacy violations (e.g., unintended capture of local forest rangers or community members).

This pipeline implements a multi-stage gated architecture prior to individual tiger re-identification:
1. **Stage 1 — Localization (Localizer)**: Locate candidate animals and objects in a full-frame capture with bounding boxes.
2. **Stage 2 — Privacy Filtering**: Suppress any detection of humans immediately—people are **never cropped, never stored, and never identified**.
3. **Stage 3 — Species Verification**: If an animal is detected, apply a real classifier with a verified taxonomic label set. Tiger re-identification is triggered **strictly** when the animal is verified as a tiger (*Panthera tigris*). If unverified or non-tiger, display `"Animal detected (species not determined)"` or the non-tiger species label, and **bypass** re-identification.
4. **Stage 4 — Gated Tiger Re-Identification**: Extract a 128-dimensional embedding from the verified tiger crop using a fine-tuned Vision Transformer (ViT-Base) and rank candidates in the reference gallery using cosine similarity.

---

## 2. Model Architectures & Licences

| Stage | Model Architecture | Parameters / Size | Pre-training Dataset | Licence | Measured CPU Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Localizer** | SSDLite320-MobileNetV3-Large | 3.4M (~13.4 MB) | COCO 2017 (80 classes) | BSD 3-Clause | ~140 ms |
| **2. Privacy Filter** | Rule-based COCO class 1 suppression | 0 params | N/A | BSD 3-Clause | < 1 ms |
| **3. Species Verifier** | MobileNetV3-Small | 2.5M (~9.8 MB) | ImageNet-1k (1,000 classes) | BSD 3-Clause | ~24 ms |
| **4. Re-ID Embedding** | ViT-Base (`vit_base_patch16_224`) + 128-dim head | 86.6M (~328.2 MB) | ImageNet-21k, fine-tuned on 74 tiger IDs | Apache-2.0 | ~180 ms |

---

## 3. Training & Embedding Methodology

- **Backbone**: Vision Transformer Base (`vit_base_patch16_224`), input size 224x224 RGB.
- **Fine-Tuning Scope**: Last 4 transformer blocks unfrozen; earlier blocks frozen.
- **Embedding Head**: `Linear(768 -> 256) -> ReLU -> Linear(256 -> 128) -> L2 Normalization`.
- **Loss Function**: Triplet Margin Loss with online semi-hard negative mining.
- **Dataset Partition**: 107 tiger identities total:
  - 74 tiger identities (70% split, random seed 42) used for fine-tuning.
  - 33 tiger identities (30% split) held out as the reference gallery for retrieval.
- **Training Trajectory**: 19 epochs trained (early stopped, patience=10). Best triplet loss achieved: `0.0054` at epoch 9.
- **Honesty Note on Loss**: Best triplet loss measures metric learning convergence on the 74-tiger training split. It does **not** measure real-world individual re-identification accuracy, field precision, or generalization to wild tigers.

---

## 4. Score Interpretations & Honesty Rules

In strict adherence to [AGENTS.md](file:///d:/Wild-ID/AGENTS.md):

1. **Cosine Similarity**:
   - Cosine similarity is a **ranking signal**, not a probability, "confidence", or verified identity. It is never labeled or presented as a probability.
   - A score above the heuristic threshold (`0.96`) indicates visual resemblance within the embedding space, not verified biological identity.
2. **Localizer Objectness Scores**:
   - Detection scores reflect heuristic box confidence from the neural network; they are ranking heuristics, not calibrated probabilities.
3. **Species Classifier Scores**:
   - Softmax output from MobileNetV3-Small reflects relative class affinity across 1,000 ImageNet categories, not calibrated probabilities.

---

## 5. Runtime Concurrency & Resource Bounds

- **Execution Mode**: CPU-only on Windows, utilizing `torch.inference_mode()` across all inference calls.
- **Thread Management**: Pinned to `torch.set_num_threads(4)` to prevent thread starvation.
- **Queue Limits**: Serialized via Gradio queue with `max_size=8`, default concurrency `1`, and explicit per-event `concurrency_limit=1`.
- **Upload Safety**:
  - Max upload size: 20 MB.
  - Decompression bomb threshold: 30,000,000 pixels (~30 MP). Images exceeding this are rejected before allocation.
  - Display downscaling: Images exceeding 2,048,000 pixels are downscaled preserving aspect ratio.
- **Per-Request Timeout**: 60-second execution timeout enforced via thread pool wrapper to prevent hung processes.
- **Auditing**: Logged to `logs/wild_id.log` (5 MB rotating, 3 backups) without image content or raw pixel representations.

---

## 6. Known Limitations & What Is Needed Before Field Claims

### Technical Limitations:
1. **Flank Asymmetry**: Tigers possess unique, non-symmetrical stripe patterns on their left and right flanks. An embedding from the left flank will produce low cosine similarity against a right-flank reference image of the exact same tiger. An automated flank classifier is required.
2. **Camera-Trap Artifacts**: Nocturnal infrared flash glare, severe motion blur, and partial flank occlusion by vegetation can distort embeddings.
3. **Heuristic Threshold**: The default threshold (`0.96`) is a configurable heuristic, not an empirically validated threshold derived from ROC / PR curves.

### Tadoba Deployment Disclaimer:
- **No Tadoba Deployment**: This system has **not** been deployed, validated, or verified in the Tadoba-Andhari Tiger Reserve (TATR).
- **Mandatory Requirements Before Field Claims**:
  1. A formal, documented, ground-truth dataset from TATR camera traps verified by field biologists.
  2. Documented False Acceptance Rate (FAR) and False Rejection Rate (FRR) curves.
  3. Flank side classification and image quality scoring.
  4. Human-in-the-loop verification protocol integrated into park management workflows.
