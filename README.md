
# Individual Identification of Tigers

**Minor Project — Department of Electronics & Telecommunication**  
---

## Overview

This system identifies individual tigers by their unique stripe patterns using image-based pattern embeddings. It fine-tunes a **Vision Transformer (ViT-Base)** with **Triplet Loss** to produce 128-dimensional embeddings per tiger image, then compares query embeddings against a reference database using **cosine similarity**.

> This is a re-identification system — not classification. New tigers can be added to the database without retraining.

---

    
```
Tiger Image (224x224)
|
Preprocessing — Resize, Normalize (ImageNet stats)
|
Fine-Tuned ViT-Base (last 4 blocks unfrozen)
CLS token -> 768-dim
|
Embedding Head: Linear(768->256) -> ReLU -> Linear(256->128) -> L2 Normalize
|
128-dim Embedding
|
Cosine Similarity vs Reference DB (threshold in src/config.py, configurable heuristic, not validated)
|
Candidate Identity (unverified) / No candidate above threshold
```


## Training Summary

| Parameter | Value |
|-----------|-------|
| Total tigers | 107 |
| Training split (70%) | 74 tigers |
| Holdout/Reference split (30%) | 33 tigers |
| Random seed | 42 |
| Epochs trained | 19 (early stopped, patience=10) |
| Best triplet loss | 0.0054 (epoch 9) |
| Embedding dimension | 128 |
| Cosine threshold | Configured in `src/config.py` (heuristic, not validated) |
| Optimizer | Adam (lr=0.0001) |
| LR Scheduler | CosineAnnealingLR |

---

## Project Structure

```

minor/
├── data/                        # Download separately (see below)
│   ├── fine_tuned_tigers/       # 74 training tigers
│   ├── reference/               # 33 holdout tigers (reference DB)
│   └── test/                    # Query images
├── models/
│   ├── tiger_finetuned_vit.pth  # Download separately (see below)
│   ├── reference_features.npz   # Atomic reference embeddings matrix
│   └── reference_features.json  # Reference metadata
├── src/
│   ├── feature_extractor.py
│   ├── matcher.py
│   └── utils.py
├── results/                     # Runtime outputs saved here
├── demo.py                      # Main Gradio UI
├── model.py                     # ViT + embedding head definition
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Adityathere06/Wild-ID.git
cd Wild-ID
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Download Required Files

### 1. Model Weights (329MB)
Download and place inside `models/`:

**[Download tiger_finetuned_vit.pth](https://drive.google.com/file/d/1GZ8hb2vX3JBfK89WqR7KZZsOHSaAdXuc/view?usp=sharing)**

### 2. Dataset
Download the zip, extract it, and place the `data/` folder in the project root:

**[Download data.zip](https://drive.google.com/file/d/1eWLyGOAd_Lt4wO07_dwzRsTHuahO5Af3/view?usp=sharing)**

```
data/
├── fine_tuned_tigers/
├── reference/
└── test/
```


---


## Run

```bash
python demo.py
```

Opens Gradio UI at `http://127.0.0.1:7861`

---

## Tech Stack

- Python 3.x
- PyTorch + torchvision
- timm (ViT-Base pretrained on ImageNet-21k)
- Gradio
- scikit-learn
- albumentations
README
```
