# Dataset Documentation: Wild-ID Tiger Re-Identification

## 1. Upstream Provenance & Licensing Disclaimer

> **IMPORTANT NOTICE PER AGENTS.MD STANDING RULES:**  
> The specific upstream dataset name, origin institution, camera trap locations, and formal license terms are **unstated** in the repository. Per repository honesty rules, these attributes are **not guessed or fabricated**.

The repository contains a curated split of 107 tiger identities organized as JPEG images in nested folders. Anyone using this repository for academic or operational purposes must verify upstream licensing terms with the original repository maintainers before public redistribution.

---

## 2. Directory Layout & Structure

The dataset is organized into three primary operational partitions under `data/`:

```
data/
├── fine_tuned_tigers/     # 74 tiger individual directories (training partition)
│   ├── tiger_001/
│   │   ├── tiger_001_001.jpg
│   │   └── ...
│   └── ...
├── reference/             # 33 tiger individual directories (reference / holdout partition)
│   ├── tiger_001/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── ...
├── test/                  # Sample query images for demonstration and batch identification
│   ├── NewTiger_not_in_database.png
│   ├── tiger_007_012.jpg
│   ├── tiger_012_009.jpg
│   └── tiger_025_032.jpg
├── pending/               # Runtime staging directory for newly submitted camera-trap images
│   └── (quarantined submissions awaiting human review)
└── tadoba-species.json    # Verified Tadoba-Andhari fauna checklist (taxonomic reference)
```

---

## 3. Partition Statistics

| Partition | Individual Identities | Purpose | Used at Inference? |
| :--- | :--- | :--- | :--- |
| `data/fine_tuned_tigers/` | 74 tigers | Fine-tuning the ViT embedding head via Triplet Loss | No |
| `data/reference/` | 33 tigers | Known reference gallery for cosine similarity retrieval | Yes (read into vector database) |
| `data/test/` | 4 images | Query validation samples (includes holdout and unseen individuals) | Yes (query inputs) |
| `data/pending/` | Dynamic | Quarantine staging for submissions awaiting reviewer approval | No (isolated until approved) |

---

## 4. Privacy, Human Filtering & Data Hygiene

In accordance with camera-trap ethics and repository safety guidelines:

1. **Human Presence Filtering**: Any camera-trap image containing human beings (e.g. forest patrol personnel, researchers, or local residents) is suppressed by the localization stage. Humans are **never cropped, never stored in reference or pending sets, and never passed to re-identification**.
2. **Safe Staging**: Images submitted through the web interface are quarantined in `data/pending/` with UUID-tagged JSON metadata. They do not enter the matching gallery until an authenticated reviewer explicitly approves the submission.
3. **Non-Destructive Deletion**: Deletion operations do not perform recursive unlinks on disk; instead, target folders are timestamped and moved to `trash/` to allow audited recovery.

---

## 5. Tadoba Fauna Checklist (`data/tadoba-species.json`)

`data/tadoba-species.json` compiles verified mammal, bird, and reptile species occurring within the **Tadoba-Andhari Tiger Reserve (TATR)**.

- **Sources**:
  - Zoological Survey of India (ZSI) Conservation Area Series (Vol. 25: Fauna of Tadoba-Andhari Tiger Reserve).
  - National Tiger Conservation Authority (NTCA) Management Effectiveness Evaluation (MEE) Tiger Reserve brief.
  - BirdLife International / Key Biodiversity Area (KBA) Factsheet 18278 (Tadoba National Park).
  - Maharashtra Forest Department official faunal checklist.
- **Model Capabilities**: Each species entry contains common and scientific names. At runtime, the system dynamically queries the loaded classifier label set to report whether the model possesses the taxonomic capability to name that species, avoiding fabricated detections.

---

## 6. Disclaimers Regarding Tadoba Deployment

- The images in `data/` **have not been verified** as originating from the Tadoba-Andhari Tiger Reserve.
- This dataset does not constitute an official tiger census database for Tadoba or any other national park.
- No performance claims regarding real-world Tadoba camera-trap deployments are made or implied.
