import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image
from speciesnet.classifier import SpeciesNetClassifier


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    if len(sys.argv) != 2:
        emit({"ready": False, "error": "Expected model directory"})
        return 2

    model_dir = Path(sys.argv[1]).resolve()
    model = SpeciesNetClassifier(str(model_dir), device="cpu")
    emit({"ready": True, "version": "4.0.3a"})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("stop"):
                break

            encoded = request.get("image")
            if not isinstance(encoded, str) or len(encoded) > 12_000_000:
                emit({"error": "Invalid or oversized image payload"})
                continue

            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")

            prepared = model.preprocess(image, bboxes=None)
            result = model.batch_predict(["memory-only"], [prepared])[0]
            classifications = result.get("classifications", {})
            classes = classifications.get("classes", [])
            scores = classifications.get("scores", [])

            if not classes or not scores:
                emit({"error": "No classification returned"})
                continue

            label = str(classes[0])
            parts = label.split(";")
            common_name = parts[-1].strip() if parts else ""
            genus = parts[4].strip() if len(parts) > 4 else ""
            species = parts[5].strip() if len(parts) > 5 else ""
            scientific_name = (
                f"{genus} {species}"
                if genus and species
                else None
            )

            emit({
                "label": common_name or None,
                "scientific_name": scientific_name,
                "score": float(scores[0]),
                "is_blank": common_name.lower() == "blank",
            })

        except Exception:
            # Do not log image bytes, filenames, or image contents.
            emit({"error": "SpeciesNet inference failed"})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        emit({"ready": False, "error": "SpeciesNet worker initialization failed"})
        raise SystemExit(1)
