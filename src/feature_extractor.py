import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TigerEmbeddingModel
from src.config import WEIGHTS_PATH

class TigerFeatureExtractor:
    """Feature extractor for tiger re-identification embeddings."""

    def __init__(self, device='cpu', weights_path=None):
        self.device = device
        weights = weights_path or WEIGHTS_PATH

        print(f"Loading fine-tuned tiger model from: {weights}")

        self.model = TigerEmbeddingModel()
        self.model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True)
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"[OK] Fine-tuned tiger model loaded successfully on {device}")

    def extract_features(self, image_input):
        """
        Extract 128-dim embedding from an image.

        Args:
            image_input: File path (str/Path) or PIL.Image.Image instance.

        Returns:
            np.ndarray of shape (128,) or None if extraction fails.
        """
        try:
            if isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            else:
                image = Image.open(image_input).convert('RGB')

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image_tensor)
            features = features.cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

# Backward compatibility alias
SnowLeopardFeatureExtractor = TigerFeatureExtractor