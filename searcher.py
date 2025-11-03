"""
Screenshot searcher - searches indexed screenshots using natural language queries
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class ScreenshotSearcher:
    """Searches indexed screenshots using semantic similarity"""

    def __init__(self, index_dir: str = "index", cache_dir: str = ".cache"):
        """Load the index and embedding model"""
        self.index_path = Path(index_dir)

        # Load metadata
        metadata_path = self.index_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found at {metadata_path}. "
                "Please run indexing first."
            )

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load embeddings
        embeddings_path = self.index_path / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {embeddings_path}. "
                "Please run indexing first."
            )

        self.embeddings = np.load(embeddings_path)

        # Load embedding model (same as used for indexing)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder=cache_dir
        )
        print("Model loaded!")

        print(f"Loaded index with {len(self.metadata)} screenshots")

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for screenshots matching the query

        Args:
            query: Natural language search query
            top_k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing search results with confidence scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Calculate similarity with all screenshots
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append((idx, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top K results
        top_results = []
        for idx, score in similarities[:top_k]:
            result = {
                "filename": self.metadata[idx]["filename"],
                "path": self.metadata[idx]["path"],
                "confidence_score": float(score),
                "ocr_text": self.metadata[idx]["ocr_text"],
                "visual_description": self.metadata[idx]["visual_description"],
            }
            top_results.append(result)

        return top_results

    def print_results(self, results: List[Dict], query: str) -> None:
        """Pretty print search results"""
        print(f"\n{'='*80}")
        print(f"Search Query: '{query}'")
        print(f"{'='*80}\n")

        if not results:
            print("No results found.")
            return

        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Filename: {result['filename']}")
            print(f"  Confidence: {result['confidence_score']:.4f}")
            print(f"  Path: {result['path']}")
            print(f"  OCR Text: {result['ocr_text'][:100]}{'...' if len(result['ocr_text']) > 100 else ''}")
            print(f"  Visual: {result['visual_description']}")
            print(f"{'-'*80}\n")


def search_screenshots(query: str, index_dir: str = "index", top_k: int = 5) -> List[Dict]:
    """Main function to search screenshots"""
    searcher = ScreenshotSearcher(index_dir)
    results = searcher.search(query, top_k)
    searcher.print_results(results, query)
    return results