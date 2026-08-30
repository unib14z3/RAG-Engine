from pathlib import Path

# H-Dex/
# ├── data/
# ├── RAG-Engine/
# │   └── src/rag_engine/core/
# └── main.py

# Project root: H-Dex/
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Source documents
DATA_DIR = PROJECT_ROOT / "data"

# Persistent ChromaDB storage
CHROMA_DIR = PROJECT_ROOT / "RAG-Engine" / "chroma_db"

# Chroma collection
COLLECTION_NAME = "sih_documents"

# Local embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
