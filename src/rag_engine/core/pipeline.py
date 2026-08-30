from pathlib import Path

from .config import TOP_K
from .ingest import build_index
from .retrieve import create_retriever


class RAGPipeline:

    def __init__(self, source_path: str | Path | None = None):
        self.index = build_index(source_path)
        self.retriever = create_retriever(
            self.index,
            TOP_K,
        )

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    def ingest_path(self, file_path: str | Path):
        """Index a PDF file or directory and refresh this pipeline's retriever.

        Content from an explicit path is added to the current Chroma collection.
        Calling this repeatedly for the same file will add duplicate chunks.
        """
        self.index = build_index(file_path)
        self.retriever = create_retriever(self.index, TOP_K)
        return self.index

    def ingestion_path(self, file_path: str | Path):
        """Alias for :meth:`ingest_path`."""
        return self.ingest_path(file_path)

    def ingestionpath(self, file_path: str | Path):
        """Backward-compatible alias for :meth:`ingest_path`."""
        return self.ingest_path(file_path)
