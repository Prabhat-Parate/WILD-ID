import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SIMILARITY_THRESHOLD, TOP_K_CANDIDATES

class TigerMatcher:
    """Matcher for tiger individual re-identification via cosine similarity ranking."""

    def __init__(self):
        self.reference_features = {}
        self.reference_names = {}

    def add_reference(self, name, features, image_path):
        """
        Add a reference individual to the database.

        Args:
            name: Name/ID of the individual tiger.
            features: Feature embedding (128-dim numpy array).
            image_path: Path to reference image.
        """
        if name not in self.reference_features:
            self.reference_features[name] = []
            self.reference_names[name] = []

        self.reference_features[name].append(features)
        self.reference_names[name].append(image_path)

    def find_match(self, query_features, threshold=SIMILARITY_THRESHOLD):
        """
        Find candidate match with highest cosine similarity.

        Args:
            query_features: Feature embedding of query image.
            threshold: Heuristic cosine similarity threshold.

        Returns:
            Tuple of (candidate_name_or_None, best_score, all_scores_dict)
        """
        if not self.reference_features:
            return None, 0.0, {}

        best_match = None
        best_score = -1.0
        all_scores = {}

        query = query_features.reshape(1, -1)

        for name, feature_list in self.reference_features.items():
            similarities = []
            for ref_features in feature_list:
                ref = ref_features.reshape(1, -1)
                sim = float(cosine_similarity(query, ref)[0][0])
                similarities.append(sim)

            max_sim = max(similarities) if similarities else -1.0
            all_scores[name] = max_sim

            if max_sim > best_score:
                best_score = max_sim
                best_match = name

        if best_score < threshold:
            return None, best_score, all_scores

        return best_match, best_score, all_scores

    def get_top_matches(self, query_features, top_k=TOP_K_CANDIDATES):
        """
        Get top K candidate matching individuals ranked by cosine similarity.

        Args:
            query_features: Feature embedding of query image.
            top_k: Number of top candidate matches to return.

        Returns:
            List of tuples (name, similarity_score)
        """
        if not self.reference_features:
            return []

        query = query_features.reshape(1, -1)
        all_scores = {}

        for name, feature_list in self.reference_features.items():
            similarities = []
            for ref_features in feature_list:
                ref = ref_features.reshape(1, -1)
                sim = float(cosine_similarity(query, ref)[0][0])
                similarities.append(sim)

            all_scores[name] = max(similarities) if similarities else -1.0

        sorted_matches = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches[:top_k]

    def get_top_candidates(self, query_features, top_k=TOP_K_CANDIDATES):
        """
        Get top K candidate matching individuals with their reference images and scores.

        Args:
            query_features: Feature embedding of query image.
            top_k: Number of candidates to return.

        Returns:
            List of dicts: [{'name': str, 'score': float, 'reference_images': list[str]}]
        """
        top_matches = self.get_top_matches(query_features, top_k=top_k)
        candidates = []
        for name, score in top_matches:
            candidates.append({
                "name": name,
                "score": score,
                "reference_images": self.get_reference_images(name)
            })
        return candidates

    def get_reference_images(self, name):
        """Get all reference image paths for a specific individual."""
        if name in self.reference_names:
            return self.reference_names[name]
        return []

    def get_statistics(self):
        """Get summary statistics of the reference database."""
        total_individuals = len(self.reference_features)
        total_images = sum(len(imgs) for imgs in self.reference_features.values())

        return {
            'total_individuals': total_individuals,
            'total_images': total_images,
            'individuals': list(self.reference_features.keys())
        }

# Backward compatibility alias
SnowLeopardMatcher = TigerMatcher