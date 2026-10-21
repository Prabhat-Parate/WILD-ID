
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights).eval()
transform = weights.transforms()
categories = weights.meta["categories"]

test_dir = Path("data/test")
extensions = {".jpg", ".jpeg", ".png"}

files = sorted(
    p for p in test_dir.iterdir()
    if p.suffix.lower() in extensions
)

for file in files:
    image = Image.open(file).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        scores = torch.softmax(model(tensor)[0], dim=0)
        values, indices = scores.topk(5)

    print(f"\n{file.name}")
    for score, index in zip(values.tolist(), indices.tolist()):
        print(f"  {categories[index]}: {score:.3f}")
