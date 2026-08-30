from pathlib import Path

import chromadb
import pymupdf
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)


def build_index(source_path: str | Path | None = None):
    Settings.embed_model = get_embedding_model()
    return ingest_documents(source_path)


def get_embedding_model():
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def load_pdfs(source_path: str | Path | None = None):
    """
    Extract text from a PDF file or every PDF below a directory.

    When no path is supplied, the configured ``DATA_DIR`` is used.
    """

    documents = []
    source = Path(source_path).expanduser().resolve() if source_path else DATA_DIR

    if source.is_file():
        pdf_files = [source] if source.suffix.lower() == ".pdf" else []
    elif source.is_dir():
        pdf_files = [
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
    else:
        raise FileNotFoundError(f"PDF source path does not exist: {source}")

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {source}"
        )

    print(f"Found {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        pdf = pymupdf.open(pdf_path)

        for page_number, page in enumerate(pdf.pages(), start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page_number": page_number,
                    },
                )
            )

        pdf.close()

    return documents


def ingest_documents(source_path: str | Path | None = None):
    """Index configured PDFs or append PDFs from an explicit path."""

    if source_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    Settings.embed_model = get_embedding_model()

    documents = load_pdfs(source_path)

    if not documents:
        raise ValueError(
            "No readable text was extracted from the PDFs."
        )

    print(f"Extracted {len(documents)} page(s) of text.")

    # Show a small sanity check
    print()
    print("Text extraction preview:")
    print(repr(documents[0].text[:200]))
    print()

    collection = get_chroma_collection()

    # Default startup uses the existing database. An explicit source path is
    # intentionally appended to the collection.
    if collection.count() > 0 and source_path is None:

        print(
            f"ChromaDB already contains "
            f"{collection.count()} chunk(s)."
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store
        )

        return index

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("Creating embeddings...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )

    print(
        f"Ingestion complete. "
        f"Stored {collection.count()} chunk(s)."
    )

    return index
