from pathlib import Path

from ..context import ContextProcessor
from .config import TOP_K
from .ingest import build_index, ingest_web as _ingest_web
from .retrieve import create_retriever


class RAGPipeline:

    def __init__(
        self,
        source_path: str | Path | list[str] | None = None,
        vlm_provider: str | None = None,
    ):
        self.vlm_provider = vlm_provider
        self.index = build_index(source_path, vlm_provider=vlm_provider)
        self.retriever = create_retriever(
            self.index,
            TOP_K,
        )
        self.context_processor = ContextProcessor()

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    def process_context(
        self,
        query: str,
        web_results: list | None = None,
        max_tokens: int = 1024,
        output_format: str = "xml",
    ):
        """Retrieves and processes context into a cleaned, weighted, and compacted string."""
        raw_rag_results = self.retrieve(query)
        return self.context_processor.process(
            rag_items=raw_rag_results,
            web_items=web_results,
            query=query,
            max_tokens=max_tokens,
            output_format=output_format,
        )

    def ingest_path(self, file_path: str | Path | list[str]):
        """Index a file, directory, or URL and refresh this pipeline's retriever.

        Content from an explicit path or URL is added to the current Chroma collection.
        Calling this repeatedly for the same source will add duplicate chunks.
        """
        self.index = build_index(file_path)
        self.retriever = create_retriever(self.index, TOP_K)
        return self.index

    def ingest_web(self, urls: str | list[str]):
        """Index web page(s) from URL(s) and refresh this pipeline's retriever."""
        self.index = _ingest_web(urls)
        self.retriever = create_retriever(self.index, TOP_K)
        return self.index

    def ingestion_path(self, file_path: str | Path | list[str]):
        """Alias for :meth:`ingest_path`."""
        return self.ingest_path(file_path)

    def ingestionpath(self, file_path: str | Path | list[str]):
        """Backward-compatible alias for :meth:`ingest_path`."""
        return self.ingest_path(file_path)
