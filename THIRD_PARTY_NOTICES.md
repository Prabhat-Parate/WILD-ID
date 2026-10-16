# Third-Party Notices & Licensing

This project incorporates and links to third-party open-source software and pre-trained neural network weights. This document summarizes third-party licenses, notices, and copyleft / commercial usage considerations.

---

## 1. Deployed Runtime Components & Licences

| Component / Library | Upstream Author / Project | Licence | Commercial Use Permitted? | Copyleft / AGPL Risk? |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch & torchvision** | Meta Platforms / PyTorch Contributors | BSD-style | Yes | None (permissive) |
| **timm (PyTorch Image Models)** | Ross Wightman | Apache-2.0 | Yes | None (permissive) |
| **Gradio** | Gradio Team / Hugging Face | Apache-2.0 | Yes | None (permissive) |
| **SSDLite320-MobileNetV3-Large** | Torchvision Team (pre-trained weights) | BSD 3-Clause | Yes | None (permissive) |
| **MobileNetV3-Small** | Torchvision Team (pre-trained weights) | BSD 3-Clause | Yes | None (permissive) |
| **ViT-Base Backbone** | Google Research / Ross Wightman | Apache-2.0 | Yes | None (permissive) |
| **Pillow (PIL Fork)** | Jeffrey A. Clark & Contributors | HPND / PIL | Yes | None (permissive) |
| **NumPy** | NumPy Developers | BSD 3-Clause | Yes | None (permissive) |
| **SciPy** | SciPy Developers | BSD 3-Clause | Yes | None (permissive) |
| **scikit-learn** | scikit-learn Developers | BSD 3-Clause | Yes | None (permissive) |
| **pandas** | pandas Development Team | BSD 3-Clause | Yes | None (permissive) |
| **pytest** | Holger Krekel & pytest-dev team | MIT | Yes | None (permissive) |
| **requests** | Kenneth Reitz & Contributors | Apache-2.0 | Yes | None (permissive) |
| **tqdm** | tqdm Contributors | MIT / MPL-2.0 | Yes | None (permissive) |

---

## 2. Evaluation of Copyleft, AGPL, and Non-Commercial Candidates

During pipeline design (documented in [MODEL_CARD.md](file:///d:/Wild-ID/MODEL_CARD.md)), multiple external animal localizer models were analyzed. We specifically evaluated and avoided models with **AGPL** or restrictive copyleft terms:

### A. Ultralytics YOLOv8 / MegaDetector v6 (AGPL-3.0)
- **Licensing Terms**: GNU Affero General Public License v3.0 (AGPL-3.0).
- **Risk Assessment**: Under AGPL-3.0 Section 13 (Remote Network Interaction), anyone interacting with the application over a network (including via Gradio web UI or REST APIs) must be provided complete corresponding source code. Commercial closed-source distribution requires purchasing proprietary enterprise licenses from Ultralytics.
- **Decision**: **Excluded from production runtime** to avoid copyleft contamination and legal uncertainty.

### B. Ultralytics YOLOv5 / MegaDetector v5 (GPL-3.0 / AGPL-3.0 upstream)
- **Licensing Terms**: While MegaDetector weights were released under MIT by Microsoft AI for Earth, the underlying YOLOv5 architecture code is licensed under GPL-3.0 / AGPL-3.0 by Ultralytics.
- **Risk Assessment**: Distribution of runtime binaries or bundled code introduces copyleft requirements.
- **Decision**: **Excluded from production runtime**.

### C. Selected Solution: SSDLite320-MobileNetV3-Large (BSD 3-Clause)
- **Licensing Terms**: BSD 3-Clause.
- **Permissiveness**: 100% commercially and academically permissive with no copyleft, viral terms, or network-distribution requirements.
- **Decision**: **Adopted as primary production localizer**.

---

## 3. Pre-Trained Neural Network Weight Notices

1. **SSDLite320-MobileNetV3-Large**:
   - Trained on COCO 2017 dataset.
   - Weights provided under PyTorch / Torchvision BSD-style license.
2. **MobileNetV3-Small (ImageNet-1k)**:
   - Trained on ImageNet-1k (ILSVRC-2012).
   - Weights provided under PyTorch / Torchvision BSD-style license.
3. **Vision Transformer (ViT-Base vit_base_patch16_224)**:
   - Pre-trained on ImageNet-21k, fine-tuned embedding head stored in `models/tiger_finetuned_vit.pth`.
   - Backbone distributed under Apache-2.0.

---

## 4. Third-Party Code Modifications

- All modifications to model heads, feature extraction scripts, and user interface logic are original contributions to this repository.
- No GPL, AGPL, or proprietary non-commercial libraries have been vendorized into the codebase.
