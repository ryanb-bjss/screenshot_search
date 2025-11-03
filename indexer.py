"""
Screenshot indexer - extracts OCR text and visual descriptions from screenshots
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ScreenshotIndexer:
    """Indexes screenshots with OCR and visual descriptions"""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize models for OCR, image captioning, and embeddings"""
        print("Loading models...")

        # OCR model
        self.ocr_reader = easyocr.Reader(['en'], gpu=True if self._has_gpu() else False)

        # Visual description model (BLIP-2)
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir
        )

        # Embedding model for semantic search
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder=cache_dir
        )

        print("Models loaded successfully!")

    @staticmethod
    def _has_gpu() -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def extract_ocr_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        result = self.ocr_reader.readtext(image_path)
        # Combine all detected text
        text = " ".join([item[1] for item in result])
        return text

    def generate_visual_description(self, image_path: str) -> str:
        """Generate visual description of image using BLIP"""
        image = Image.open(image_path).convert('RGB')

        # Conditional image captioning
        text = "a screenshot showing"
        inputs = self.blip_processor(image, text, return_tensors="pt")

        # Generate caption
        out = self.blip_model.generate(**inputs, max_length=100)
        description = self.blip_processor.decode(out[0], skip_special_tokens=True)

        return description

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding vector from text"""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def process_screenshot(self, image_path: str) -> Dict:
        """Process a single screenshot"""
        print(f"Processing: {os.path.basename(image_path)}")

        # Extract OCR text
        ocr_text = self.extract_ocr_text(image_path)

        # Generate visual description
        visual_desc = self.generate_visual_description(image_path)

        # Combine for embedding
        combined_text = f"OCR: {ocr_text} | Visual: {visual_desc}"

        # Create embedding
        embedding = self.create_embedding(combined_text)

        return {
            "path": image_path,
            "filename": os.path.basename(image_path),
            "ocr_text": ocr_text,
            "visual_description": visual_desc,
            "combined_text": combined_text,
            "embedding": embedding
        }

    def index_directory(self, screenshots_dir: str, output_dir: str = "index") -> None:
        """Index all screenshots in a directory"""
        screenshots_path = Path(screenshots_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        image_files = [
            str(f) for f in screenshots_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No images found in {screenshots_dir}")
            return

        print(f"Found {len(image_files)} screenshots to index")

        # Process all screenshots
        index_data = []
        embeddings = []

        for image_path in tqdm(image_files, desc="Indexing screenshots"):
            try:
                result = self.process_screenshot(image_path)

                # Store embedding separately
                embeddings.append(result["embedding"])

                # Store metadata (without embedding)
                metadata = {k: v for k, v in result.items() if k != "embedding"}
                index_data.append(metadata)

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        # Save metadata as JSON
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        # Save embeddings as numpy array
        embeddings_path = output_path / "embeddings.npy"
        embeddings_array = np.array(embeddings)
        np.save(embeddings_path, embeddings_array)

        print(f"\nIndexing complete!")
        print(f"Indexed {len(index_data)} screenshots")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Embeddings saved to: {embeddings_path}")


def index_screenshots(screenshots_dir: str, output_dir: str = "index") -> None:
    """Main function to index screenshots"""
    indexer = ScreenshotIndexer()
    indexer.index_directory(screenshots_dir, output_dir)
