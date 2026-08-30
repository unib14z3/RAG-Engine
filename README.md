# RAG Engine

A Python package for indexing PDF documents in ChromaDB and retrieving the
most relevant passages for a query.

## Install

From this directory, install the package in editable mode:

```bash
python -m pip install -e .
```

The package dependencies are declared in `pyproject.toml`.

## Package layout

```text
RAG-Engine/
├── chroma_db/                 # Generated persistent vector database
├── pyproject.toml
└── src/
    └── rag_engine/
        ├── __init__.py        # Public API
        └── core/
            ├── config.py
            ├── ingest.py
            ├── pipeline.py
            └── retrieve.py
```

Import the package as `rag_engine`; the repository name contains a hyphen and
therefore cannot be used as a Python import name.

## Usage

```python
from rag_engine import RAGPipeline

pipeline = RAGPipeline("/path/to/document.pdf")

results = pipeline.retrieve("What does the document say about time travel?")
for result in results:
    print(result.text)
```

`build_index()` extracts text from all PDFs in the configured data directory,
chunks the text, creates embeddings, and saves them to ChromaDB. If the Chroma
collection already contains data, it is reused.

The constructor accepts a PDF file or a directory of PDFs. To add another PDF
or directory after the pipeline has been created, use:

```python
pipeline.ingest_path("/path/to/document.pdf")
# or: pipeline.ingest_path("/path/to/pdf-directory")
```

The provided content is added to the existing collection and the pipeline is
refreshed automatically. Do not ingest the same file repeatedly unless you
intend to store duplicate chunks. `ingestion_path()` and `ingestionpath()` are
also supported aliases.

## Configuration

Edit `src/rag_engine/core/config.py` to configure:

- `DATA_DIR` — directory containing source PDFs
- `CHROMA_DIR` — directory for the persistent ChromaDB collection
- `COLLECTION_NAME` — ChromaDB collection name
- `EMBEDDING_MODEL` — Hugging Face model ID or local model directory
- `CHUNK_SIZE` and `CHUNK_OVERLAP` — chunking settings
- `TOP_K` — number of passages returned per query

The data and database locations can also be configured without editing source
code:

```bash
RAG_ENGINE_DATA_DIR=/path/to/pdfs \
RAG_ENGINE_CHROMA_DIR=/path/to/chroma_db \
python your_script.py
```

If `RAG_ENGINE_DATA_DIR` is not set, the engine uses `data/` beneath the
directory where the application is run. `RAG_ENGINE_CHROMA_DIR` similarly
defaults to `chroma_db/` there.

## Local and offline embeddings

Embeddings are generated locally. The default model,
`BAAI/bge-small-en-v1.5`, is downloaded from Hugging Face on its first use and
then loaded from the local cache on later runs.

After the model has been downloaded, require offline mode with:

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

To avoid using a user-level model cache, download the model files into a
directory under your control and set `EMBEDDING_MODEL` to that local path.

## Rebuild the vector index

To index updated PDFs from scratch, remove the contents of the configured
`CHROMA_DIR`, then call `build_index()` again. This removes only the generated
vector database, not the source PDF files.
