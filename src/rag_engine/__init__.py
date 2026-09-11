"""Public API for the retrieval-augmented generation engine."""

from .context import (
    ContextCleaner,
    ContextCompressor,
    ContextFormatter,
    ContextProcessor,
    ContextScorer,
    ProcessedContextResult,
    WeightedContextItem,
)

__all__ = (
    "RAGPipeline",
    "build_index",
    "ingest_web",
    "ContextProcessor",
    "ContextCleaner",
    "ContextScorer",
    "ContextCompressor",
    "ContextFormatter",
    "ProcessedContextResult",
    "WeightedContextItem",
)


def __getattr__(name: str):
    if name in ("RAGPipeline", "build_index", "ingest_web"):
        from .core import ingest, pipeline

        if name == "RAGPipeline":
            return pipeline.RAGPipeline
        elif name == "build_index":
            return ingest.build_index
        elif name == "ingest_web":
            return ingest.ingest_web
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
