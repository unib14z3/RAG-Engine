import os
from pathlib import Path

# Installed packages live in site-packages, so do not derive project paths
# from this module's location. By default, use the directory the application
# is run from; deployments can override each path independently.
PROJECT_ROOT = Path(
    os.environ.get("RAG_ENGINE_PROJECT_ROOT", Path.cwd())
).resolve()

# Source documents
DATA_DIR = Path(
    os.environ.get("RAG_ENGINE_DATA_DIR", PROJECT_ROOT / "data")
).resolve()

# Persistent ChromaDB storage
CHROMA_DIR = Path(
    os.environ.get("RAG_ENGINE_CHROMA_DIR", PROJECT_ROOT / "chroma_db")
).resolve()

# Chroma collection
COLLECTION_NAME = "sih_documents"

# Local embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# Extracted Images directory
EXTRACTED_IMAGES_DIR = Path(
    os.environ.get("RAG_ENGINE_IMAGES_DIR", DATA_DIR / "extracted_images")
).resolve()

# VLM Settings
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "auto").lower()  # "auto", "gemini", or "lmstudio"
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.6-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:2000/v1")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "google/gemma-4-e4b")
VLM_ENABLED = os.environ.get("VLM_ENABLED", "true").lower() in ("true", "1", "yes")
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "180.0"))

