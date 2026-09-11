from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core import Document


class BaseIngester(ABC):
    """Abstract base class for document ingesters."""

    @abstractmethod
    def can_handle(self, source: str | Path) -> bool:
        """Return True if this ingester can process the given source."""
        pass

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """Load and extract text content from the source into Documents."""
        pass
