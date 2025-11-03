# Screenshot Search App

Natural language search for screenshots using OCR text extraction and visual descriptions powered by open source AI models.

## Features

- **OCR Text Extraction**: Extracts all visible text from screenshots using EasyOCR
- **Visual Descriptions**: Generates semantic descriptions of screenshot content using BLIP-2
- **Natural Language Search**: Search using queries like "error message about auth" or "screenshot with blue button"
- **Confidence Scores**: Returns top 5 matches with similarity confidence scores
- **100% Open Source**: Uses only open source models (no API keys required)

## Architecture

### Tech Stack

- **OCR**: EasyOCR (multilingual text extraction)
- **Vision Model**: BLIP-2 (Salesforce - image captioning)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2 for semantic search)
- **Search**: Cosine similarity with numpy
- **Storage**: JSON metadata + numpy arrays

### How It Works

1. **Indexing Phase**:
   - Scans screenshot directory
   - Extracts OCR text from each image
   - Generates visual description using BLIP-2
   - Creates semantic embeddings from combined text
   - Stores index to disk

2. **Search Phase**:
   - Takes natural language query
   - Generates query embedding
   - Calculates cosine similarity with all indexed screenshots
   - Returns top 5 matches with confidence scores

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: First run will download models (~500MB). Models are cached in `.cache/` directory.

## Usage

### 1. Index Screenshots

Index all screenshots in a directory:

```bash
python main.py index /path/to/screenshots
```

With custom output directory:

```bash
python main.py index /path/to/screenshots --output-dir my_index
```

This creates an index directory with:
- `metadata.json` - Screenshot metadata (paths, OCR text, descriptions)
- `embeddings.npy` - Semantic embeddings for search

### 2. Search Screenshots

Search with natural language queries:

```bash
python main.py search "error message about authentication"
```

With custom index directory and result count:

```bash
python main.py search "blue button" --index-dir my_index --top-k 10
```

### Example Queries

- "error message about auth"
- "screenshot with blue button"
- "code editor window"
- "login form"
- "dashboard with charts"
- "terminal showing git commands"

## Output Format

Search results include:

```
Result 1:
  Filename: screenshot_2024_01_15.png
  Confidence: 0.8542
  Path: /path/to/screenshot.png
  OCR Text: Login failed: Invalid credentials
  Visual: a screenshot showing an error dialog with red text and a close button
```

## Project Structure

```
screenshot_search/
├── main.py           # CLI interface
├── indexer.py        # Screenshot indexing (OCR + visual descriptions)
├── searcher.py       # Semantic search functionality
├── requirements.txt  # Python dependencies
├── index/           # Generated index (created after indexing)
│   ├── metadata.json
│   └── embeddings.npy
└── .cache/          # Model cache (created on first run)
```

## Performance Notes

- **GPU**: Automatically uses GPU if available (significantly faster)
- **CPU**: Works on CPU but indexing will be slower
- **Indexing**: ~5-10 seconds per screenshot on CPU, ~1-2 seconds on GPU
- **Search**: Very fast (<1 second for thousands of screenshots)

## Limitations

- Currently supports English OCR (can be extended to other languages)
- Visual descriptions are general-purpose (can be fine-tuned for specific domains)
- Requires re-indexing when adding new screenshots

## Future Enhancements

- Incremental indexing for new screenshots
- Multi-language OCR support
- Vector database integration for larger datasets
- Web interface
- Batch processing optimization