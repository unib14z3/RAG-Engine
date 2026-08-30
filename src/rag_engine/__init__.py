"""Public API for the H-Dex retrieval-augmented generation engine."""

from .core.ingest import build_index
from .core.pipeline import RAGPipeline

__all__ = ("RAGPipeline", "build_index")
