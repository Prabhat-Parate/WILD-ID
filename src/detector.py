"""Animal detection and species verification pipeline.

Implements:
1. Real object localizer (SSDLite320-MobileNetV3) with bounded detections.
2. Privacy enforcement: people are ignored, never cropped or stored.
3. Species verification: real classifier (MobileNetV3-Small ImageNet-1k).
4. Re-ID gate: tiger re-identification runs strictly on crops verified as tiger.
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torchvision.models.detection as detection_models
import torchvision.models as classification_models

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BASE_DIR

COCO_ANIMAL_CATEGORIES = {
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
}

TADOBA_SPECIES_JSON_PATH = str(BASE_DIR / "data" / "tadoba-species.json")


class AnimalDetector:
    """Real object localizer using SSDLite320-MobileNetV3 on CPU."""

    def __init__(self, device='cpu'):
        self.device = device
        weights = detection_models.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = detection_models.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        self.categories = weights.meta['categories']

    def detect(self, image: Image.Image, score_threshold=0.25, max_detections=10):
        """
        Detect objects in an image.

        Args:
            image: PIL.Image instance.
            score_threshold: Minimum detection score threshold.
            max_detections: Maximum detections to return (bounded count).

        Returns:
            list of dicts: [{'box': [x1, y1, x2, y2], 'score': float, 'category': str, 'label': str}]
        """
        rgb_img = image.convert('RGB')
        tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        results = []
        for box, score, label_idx in zip(boxes, scores, labels):
            score_val = float(score)
            if score_val < score_threshold:
                continue

            label_name = self.categories[label_idx] if label_idx < len(self.categories) else f"class_{label_idx}"

            if label_name == 'person':
                category = 'person'
            elif label_name in COCO_ANIMAL_CATEGORIES:
                category = 'animal'
            else:
                category = 'other'

            # Scale box coordinates from 320x320 back to original image dimensions
            orig_w, orig_h = rgb_img.size
            # SSDLite transform resizes to (320, 320)
            x1 = max(0.0, float(box[0]) * (orig_w / 320.0))
            y1 = max(0.0, float(box[1]) * (orig_h / 320.0))
            x2 = min(float(orig_w), float(box[2]) * (orig_w / 320.0))
            y2 = min(float(orig_h), float(box[3]) * (orig_h / 320.0))

            results.append({
                'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                'score': round(score_val, 4),
                'raw_label': label_name,
                'category': category
            })

            if len(results) >= max_detections:
                break

        return results


class SpeciesClassifier:
    """Real species classifier using MobileNetV3-Small on ImageNet-1k."""

    def __init__(self, device='cpu'):
        self.device = device
        weights = classification_models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = classification_models.mobilenet_v3_small(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        self.categories = weights.meta['categories']

    def classify_crop(self, crop_image: Image.Image):
        """
        Classify an animal crop.

        Returns:
            dict: {'label': str, 'score': float, 'is_tiger': bool, 'class_index': int}
        """
        rgb_crop = crop_image.convert('RGB')
        # Guard against zero or degenerate dimensions
        if rgb_crop.width < 10 or rgb_crop.height < 10:
            return {
                'label': "Crop too small",
                'score': 0.0,
                'is_tiger': False,
                'class_index': -1
            }

        tensor = self.transforms(rgb_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_score, top_idx = torch.topk(probs, k=1)

        pred_idx = int(top_idx[0].item())
        pred_score = float(top_score[0].item())
        pred_label = self.categories[pred_idx]

        # Exact tiger class in ImageNet-1k is index 292 ('tiger')
        is_tiger = (pred_idx == 292 or pred_label.lower().strip() == 'tiger')

        return {
            'label': pred_label,
            'score': round(pred_score, 4),
            'is_tiger': is_tiger,
            'class_index': pred_idx
        }

    def can_name_species(self, common_name: str, scientific_name: str = "") -> tuple[bool, str]:
        """
        Compute whether this model's label set can name a given species.

        Returns:
            (can_name: bool, matching_label: str)
        """
        c_lower = common_name.lower().strip()
        s_lower = scientific_name.lower().strip()

        # Specific verified taxonomic mappings to ImageNet-1k categories
        if "tiger" in c_lower and "panthera tigris" in s_lower:
            return True, "tiger"
        if "leopard" in c_lower and "panthera pardus" in s_lower:
            return True, "leopard"
        if "sloth bear" in c_lower or "melursus ursinus" in s_lower:
            return True, "sloth bear"
        if "dhole" in c_lower or "cuon alpinus" in s_lower:
            return True, "dhole"
        if "hyena" in c_lower and "hyaena" in s_lower:
            return True, "hyena"
        if "wild boar" in c_lower or "sus scrofa" in s_lower:
            return True, "wild boar"
        if "porcupine" in c_lower or "hystrix indica" in s_lower:
            return True, "porcupine"
        if "hare" in c_lower or "lepus nigricollis" in s_lower:
            return True, "hare"
        if "langur" in c_lower or "semnopithecus" in s_lower:
            return True, "langur"
        if "peafowl" in c_lower or "pavo cristatus" in s_lower:
            return True, "peacock"
        if "python" in c_lower or "python molurus" in s_lower:
            return True, "rock python"
        if "cobra" in c_lower or "naja naja" in s_lower:
            return True, "Indian cobra"

        # Check exact matches against model categories, rejecting partial token collisions
        for cat in self.categories:
            synonyms = [s.strip().lower() for s in cat.split(',')]
            for syn in synonyms:
                if syn == c_lower or (s_lower and syn == s_lower):
                    return True, cat

        return False, "Not in model label set"


def get_tadoba_species_with_capabilities(classifier: SpeciesClassifier = None):
    """
    Load data/tadoba-species.json and compute for each species whether the loaded model can name it.
    Does NOT state a total species count.
    """
    if not os.path.exists(TADOBA_SPECIES_JSON_PATH):
        return []

    with open(TADOBA_SPECIES_JSON_PATH, 'r', encoding='utf-8') as f:
        species_list = json.load(f)

    annotated = []
    for item in species_list:
        entry = dict(item)
        if classifier is not None:
            can_name, matched_label = classifier.can_name_species(
                item.get('common_name', ''),
                item.get('scientific_name', '')
            )
            entry['model_can_name'] = can_name
            entry['matched_model_label'] = matched_label
        else:
            entry['model_can_name'] = False
            entry['matched_model_label'] = "No model loaded"
        annotated.append(entry)

    return annotated


def process_frame(
    image: Image.Image,
    detector: AnimalDetector,
    classifier: SpeciesClassifier = None,
    reid_extractor = None,
    reid_matcher = None,
    score_threshold = 0.25,
    max_detections = 10
):
    """
    Execute complete localization, privacy filtering, species verification,
    and gated tiger re-identification on a camera-trap frame.

    Returns:
        dict: {
            'annotated_image': PIL.Image,
            'detections': list of dicts,
            'tiger_crops': list of PIL.Image,
            'summary_text': str
        }
    """
    rgb_img = image.convert('RGB')
    raw_detections = detector.detect(rgb_img, score_threshold=score_threshold, max_detections=max_detections)

    annotated = rgb_img.copy()
    draw = ImageDraw.Draw(annotated)

    processed_detections = []
    tiger_crops = []
    summary_lines = []

    if not raw_detections:
        summary_lines.append("No animals or objects detected in frame above score threshold.")
        return {
            'annotated_image': annotated,
            'detections': [],
            'tiger_crops': [],
            'summary_text': "\n".join(summary_lines)
        }

    summary_lines.append(f"Detections in frame ({len(raw_detections)} bounded items):")
    summary_lines.append("─" * 55)

    for idx, det in enumerate(raw_detections, 1):
        box = det['box']
        det_score = det['score']
        category = det['category']
        raw_label = det['raw_label']

        x1, y1, x2, y2 = box

        if category == 'person':
            # PRIVACY ENFORCEMENT: Never crop, never store, never re-identify people
            display_label = "Person [Ignored per privacy policy]"
            box_color = "#e74c3c"  # red
            summary_lines.append(f"  {idx}. [PERSON DETECTED] Detection score: {det_score:.2f} (ranking score, not probability) — Ignored per privacy policy.")

            # Draw box with warning color, no crop stored
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            draw.text((x1 + 4, y1 + 4), f"Person (Ignored) | Score: {det_score:.2f}", fill=box_color)

            processed_detections.append({
                'index': idx,
                'box': box,
                'score': det_score,
                'label': display_label,
                'category': 'person',
                'is_tiger': False,
                'reid_result': None
            })

        elif category == 'animal':
            crop = rgb_img.crop((x1, y1, x2, y2))

            if classifier is not None:
                cls_result = classifier.classify_crop(crop)
                species_label = cls_result['label']
                cls_score = cls_result['score']
                is_tiger = cls_result['is_tiger']

                if is_tiger:
                    display_label = f"Tiger (Panthera tigris)"
                    box_color = "#2ecc71"  # green
                    tiger_crops.append(crop)

                    reid_text = "Not evaluated"
                    if reid_extractor is not None and reid_matcher is not None:
                        # Gated Re-ID runs strictly on tiger crops!
                        features = reid_extractor.extract_features(crop)
                        if features is not None:
                            best_match, best_score, _ = reid_matcher.find_match(features)
                            top_5 = reid_matcher.get_top_matches(features, top_k=5)
                            reid_text = f"Top candidate: {best_match or 'No candidate above threshold'} (Cosine sim: {best_score:.4f})"

                    summary_lines.append(
                        f"  {idx}. [TIGER VERIFIED] {display_label}\n"
                        f"     Detector score : {det_score:.2f} (ranking score, not probability)\n"
                        f"     Classifier     : {species_label} (score: {cls_score:.2f})\n"
                        f"     Re-ID Result   : {reid_text}"
                    )

                    processed_detections.append({
                        'index': idx,
                        'box': box,
                        'score': det_score,
                        'label': display_label,
                        'species_score': cls_score,
                        'category': 'animal',
                        'is_tiger': True,
                        'reid_result': reid_text
                    })

                else:
                    display_label = f"Non-tiger animal ({species_label})"
                    box_color = "#f39c12"  # amber
                    summary_lines.append(
                        f"  {idx}. [NON-TIGER ANIMAL] {display_label}\n"
                        f"     Detector score : {det_score:.2f} (ranking score, not probability)\n"
                        f"     Classifier     : {species_label} (score: {cls_score:.2f})\n"
                        f"     Re-ID Action   : Bypassed (tiger re-identification not applicable)."
                    )
                    processed_detections.append({
                        'index': idx,
                        'box': box,
                        'score': det_score,
                        'label': display_label,
                        'species_score': cls_score,
                        'category': 'animal',
                        'is_tiger': False,
                        'reid_result': "Bypassed (non-tiger animal)"
                    })

            else:
                display_label = "Animal detected (species not determined)"
                box_color = "#3498db"  # blue
                summary_lines.append(
                    f"  {idx}. [ANIMAL DETECTED] {display_label}\n"
                    f"     Detector score : {det_score:.2f} (ranking score, not probability)\n"
                    f"     Re-ID Action   : Bypassed (requires verified tiger label)."
                )
                processed_detections.append({
                    'index': idx,
                    'box': box,
                    'score': det_score,
                    'label': display_label,
                    'category': 'animal',
                    'is_tiger': False,
                    'reid_result': "Bypassed (species not determined)"
                })

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            draw.text((x1 + 4, y1 + 4), f"{display_label} | Score: {det_score:.2f}", fill=box_color)

        else:
            display_label = f"Object ({raw_label})"
            box_color = "#95a5a6"
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
            draw.text((x1 + 4, y1 + 4), f"{raw_label} | Score: {det_score:.2f}", fill=box_color)
            processed_detections.append({
                'index': idx,
                'box': box,
                'score': det_score,
                'label': display_label,
                'category': 'other',
                'is_tiger': False,
                'reid_result': None
            })

    return {
        'annotated_image': annotated,
        'detections': processed_detections,
        'tiger_crops': tiger_crops,
        'summary_text': "\n".join(summary_lines)
    }
