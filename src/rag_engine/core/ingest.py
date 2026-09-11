from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn

from ..ingest import PDFIngester, load_documents, ingest_web as _ingest_web
from ..utils.console import (
    console,
    print_banner,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)


def build_index(
    source_path: str | Path | list[str] | None = None,
    vlm_provider: str | None = None,
):
    print_banner(source_path, vlm_provider=vlm_provider)
    Settings.embed_model = get_embedding_model()
    return ingest_documents(source_path, vlm_provider=vlm_provider)


def get_embedding_model():
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def load_pdfs(source_path: str | Path | None = None) -> list[Document]:
    """Backward-compatible wrapper to extract text from PDFs using PDFIngester."""
    target_path = source_path if source_path is not None else DATA_DIR
    return PDFIngester().load(target_path)


def ingest_web(urls: str | list[str]):
    """Ingest web pages from URL(s) into the Chroma vector store index."""
    return build_index(urls)


def ingest_documents(
    source_path: str | Path | list[str] | None = None,
    vlm_provider: str | None = None,
):
    """Index configured documents (PDF, text, web URLs) or append from source_path."""

    if source_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    Settings.embed_model = get_embedding_model()

    # Step 1: Document Discovery & Extraction
    print_step(1, 4, "Extracting Text & Embedded Visual Content")
    if source_path is None:
        documents = load_documents(DATA_DIR, vlm_provider=vlm_provider)
    else:
        documents = load_documents(source_path, vlm_provider=vlm_provider)

    if not documents:
        raise ValueError("No readable text or visual content was extracted from the provided source.")

    text_docs = [d for d in documents if d.metadata.get("content_type") != "image_caption"]
    image_docs = [d for d in documents if d.metadata.get("content_type") == "image_caption"]

    print_success(f"Extracted {len(text_docs)} text section(s) + {len(image_docs)} visual caption section(s).")

    collection = get_chroma_collection()

    # Default startup uses existing database if source_path is None
    if collection.count() > 0 and source_path is None:
        print_info(f"ChromaDB already contains {collection.count()} indexed chunk(s). Ready for query.")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store)

    # Step 2 & 3: Sentence Chunking & Text Splitting
    print_step(2, 4, f"Splitting Documents into Chunks ({CHUNK_SIZE} tokens, overlap {CHUNK_OVERLAP})")
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print_success(f"Generated {len(nodes)} document node(s) for vector indexing.")

    # Step 4: Embedding Computation & Vector Storage
    print_step(3, 4, f"Computing Embeddings using {EMBEDDING_MODEL}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing nodes into ChromaDB...", total=len(nodes))
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False,
        )
        progress.update(task, completed=len(nodes))

    print_step(4, 4, "Vector Store Sync & Pipeline Ready")
    print_success(f"Ingestion Complete! ChromaDB contains {collection.count()} indexed node(s).")
    console.print()

    return index
