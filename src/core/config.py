from pathlib import Path

# H-Dex/
# ├── data/
# ├── RAG_Engine/
# │   └── src/
# │       └── core/
# └── main.py

# Project root: H-Dex/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Source documents
DATA_DIR = PROJECT_ROOT / "data"

# Persistent ChromaDB storage
CHROMA_DIR = PROJECT_ROOT / "RAG_Engine" / "chroma_db"

# Chroma collection
COLLECTION_NAME = "sih_documents"

# Local embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

