"""
Streamlit UI for Screenshot Search
Natural language search interface for indexed screenshots
"""
import streamlit as st
from pathlib import Path
from PIL import Image
from searcher import ScreenshotSearcher


def init_searcher(index_dir: str = "index", cache_dir: str = ".cache"):
    """Initialize the searcher with caching"""
    return ScreenshotSearcher(index_dir=index_dir, cache_dir=cache_dir)


def display_result(result: dict, rank: int):
    """Display a single search result"""
    st.markdown(f"### Result {rank}")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Display the screenshot
        try:
            image = Image.open(result['path'])
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        # Display metadata
        st.metric("Confidence Score", f"{result['confidence_score']:.4f}")
        st.markdown(f"**Filename:** `{result['filename']}`")

        with st.expander("OCR Text", expanded=False):
            if result['ocr_text']:
                st.text(result['ocr_text'])
            else:
                st.info("No text detected")

        with st.expander("Visual Description", expanded=True):
            st.write(result['visual_description'])

    st.divider()


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Screenshot Search",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Screenshot Search")
    st.markdown("Search your indexed screenshots using natural language queries")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        index_dir = st.text_input("Index Directory", value="index")
        cache_dir = st.text_input("Cache Directory", value=".cache")
        top_k = st.slider("Number of Results", min_value=1, max_value=10, value=5)

        st.divider()
        st.markdown("### About")
        st.info(
            "This app searches through indexed screenshots using semantic similarity. "
            "Make sure you've indexed your screenshots first using the CLI."
        )

    # Check if index exists
    index_path = Path(index_dir)
    if not index_path.exists() or not (index_path / "metadata.json").exists():
        st.error(
            f"Index not found at `{index_dir}`. "
            "Please run the indexing command first:\n\n"
            f"```bash\npython main.py index <screenshots_dir>\n```"
        )
        st.stop()

    # Initialize searcher with caching
    @st.cache_resource
    def load_searcher(index_dir, cache_dir):
        """Load the searcher once and cache it"""
        return init_searcher(index_dir, cache_dir)

    try:
        with st.spinner("Loading search index..."):
            searcher = load_searcher(index_dir, cache_dir)

        # Display index info
        st.success(f"Loaded index with {len(searcher.metadata)} screenshots")

    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.stop()

    # Search interface
    st.markdown("---")
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'screenshot with code', 'image with a graph', 'meeting notes'",
        help="Use natural language to describe what you're looking for"
    )

    search_button = st.button("Search", type="primary", use_container_width=True)

    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                results = searcher.search(query, top_k=top_k)

                if results:
                    st.markdown(f"## Search Results for: *'{query}'*")
                    st.markdown(f"Found {len(results)} results")
                    st.markdown("---")

                    # Display results
                    for i, result in enumerate(results, 1):
                        display_result(result, i)
                else:
                    st.warning("No results found. Try a different query.")

            except Exception as e:
                st.error(f"Error during search: {e}")

    elif search_button and not query:
        st.warning("Please enter a search query")


if __name__ == "__main__":
    main()
