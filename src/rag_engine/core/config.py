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
