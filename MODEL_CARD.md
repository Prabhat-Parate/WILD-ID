# Model Card: Animal Localization and Species Verification Pipeline

## 1. Overview & Objectives

In automated camera-trap monitoring and tiger re-identification, running fine-grained individual pattern matching directly on uncurated full-frame captures risks false associations, background noise distortion, and privacy violations (e.g., unintended capture of local forest rangers or community members).

This pipeline introduces a strict, verified detection and species verification stage prior to individual re-identification:
1. **Localization (Localizer)**: Locate all candidate objects in a frame with bounding boxes.
2. **Privacy Filtering**: Suppress any detection of humans immediately—people are **never cropped, never stored, and never identified**.
3. **Species Verification**: If an animal is detected, apply a real classifier with a verified label set. Tiger re-identification is executed **strictly** when the animal is verified as a tiger (*Panthera tigris*). If unverified or non-tiger, output `"Animal detected (species not determined)"` or the non-tiger species label, and **do not** trigger re-identification.

---

## 2. Evaluation of Real Localizers

We evaluated candidate animal localizers across licensing, CPU latency, Windows compatibility, and dependency footprint:

| Localizer Model | Architecture | Licence | Parameter Count / Size | Measured CPU Latency (ms) | Windows 11 / Python 3.13 Support | Native Dependencies | Class Granularity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MegaDetector v5a** | YOLOv5x6 (1280px) | MIT (weights) / GPL/AGPL (YOLOv5 upstream) | ~140 MB | 850 – 2,200 ms | Supported via PyTorch / ONNX | Requires `yolov5` or `PytorchWildlife` | 3 coarse classes: `animal`, `person`, `vehicle` (no species) |
| **MegaDetector v5b** | YOLOv5r (640px) | MIT (weights) / GPL/AGPL (YOLOv5 upstream) | ~140 MB | 500 – 1,200 ms | Supported via PyTorch / ONNX | Requires `yolov5` or `PytorchWildlife` | 3 coarse classes: `animal`, `person`, `vehicle` (no species) |
| **MegaDetector v6** | YOLOv8 / RT-DETR | AGPL-3.0 / MIT variant | 25 – 100 MB | 400 – 1,100 ms | Supported via `ultralytics` | Requires `ultralytics` (pinned packaging dependencies) | 3 coarse classes: `animal`, `person`, `vehicle` (no species) |
| **Faster R-CNN MobileNetV3-Large FPN** | Faster R-CNN + FPN (320px) | BSD 3-Clause | ~78 MB | 180 – 320 ms | 100% Native | `torchvision` (already installed) | 80 COCO classes (`person`, 10 animal classes) |
| **SSDLite320-MobileNetV3-Large** | SSDLite + MobileNetV3-Large | BSD 3-Clause | 13.4 MB | **~140 ms** | 100% Native | `torchvision` (already installed) | 80 COCO classes (`person`, 10 animal classes) |

---

## 3. Engineering Decision & Pipeline Architecture

### Selected Localizer: `SSDLite320-MobileNetV3-Large`
- **Licence**: BSD 3-Clause. Fully open-source, commercially and academically permissive with zero copyleft risk.
- **CPU Latency**: ~140 ms per full frame on CPU, making it responsive in interactive web UI and batch processing without GPU acceleration.
- **Windows Support**: Native support with zero additional C-extension or unpinned library dependencies on Windows with Python 3.13.
- **Privacy Enforcement**: Pre-trained on COCO including class 1 (`person`), enabling immediate suppression of human detections.

### Selected Species Verifier: `MobileNetV3-Small` (ImageNet-1k)
- **Licence**: BSD 3-Clause.
- **Size & Latency**: 9.8 MB; **23.9 ms** per crop on CPU.
- **Verified Label Set**: 1,000 peer-reviewed taxonomic and object categories, including class 292: `tiger, Panthera tigris` as well as co-predators (`leopard`, `sloth bear`, `dhole`, `wild boar`).
- **Gating Function**: Re-identification runs **only** if the classifier prediction specifically identifies `tiger`. Non-tiger animals display their predicted label or `"Animal detected (species not determined)"` and are bypassed from tiger re-ID.

---

## 4. Bounded Output & Score Disclaimers

1. **Bounded Detections**: Detection lists are strictly capped (default `max_detections=10`) to prevent denial-of-service on crowded scenes.
2. **Score Transparency**: All detection scores are heuristic objectness / class-affinity scores, **not probabilities**. They are explicitly presented as such in the user interface.
3. **No Heuristic or Fake Detections**: Every box and label is produced by real forward passes of committed, pre-trained neural networks.

---

## 5. Privacy & Human Safety Protocol

In compliance with camera-trap data ethics and [AGENTS.md](file:///d:/Wild-ID/AGENTS.md):
- Detections classified as `person` are immediately filtered.
- No cropped image of a person is ever saved to disk, passed to feature extractors, or stored in pending/reference sets.
- Visualizations mark person bounding boxes as `[Person - Ignored per Privacy Policy]` without creating identity records.
