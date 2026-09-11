from pathlib import Path

from llama_index.core import Document

from ..vlm import ImageCaptioner
from .base import BaseIngester
from .image import ImageIngester
from .pdf import PDFIngester
from .text import TextIngester
from .web import WebIngester, ingest_web


class IngestionRouter:
    """Dispatches sources (URLs, single files, or directories) to appropriate ingesters."""

    def __init__(
        self,
        captioner: ImageCaptioner | None = None,
        vlm_provider: str | None = None,
    ):
        if captioner is not None:
            self.captioner = captioner
        else:
            self.captioner = ImageCaptioner(provider=vlm_provider) if vlm_provider else ImageCaptioner()

        self.ingesters: list[BaseIngester] = [
            PDFIngester(captioner=self.captioner),
            ImageIngester(captioner=self.captioner),
            TextIngester(),
            WebIngester(),
        ]

    def load(self, source: str | Path | None = None) -> list[Document]:
        if source is None:
            return []

        source_str = str(source).strip()
        if source_str.startswith(("http://", "https://")):
            return ingest_web(source_str)

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")

        if path.is_file():
            for ingester in self.ingesters:
                if ingester.can_handle(path):
                    return ingester.load(path)
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if path.is_dir():
            documents = []
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    for ingester in self.ingesters:
                        if ingester.can_handle(file_path):
                            documents.extend(ingester.load(file_path))
                            break
            return documents

        raise ValueError(f"Invalid source: {source}")


def load_documents(
    source: str | Path | list[str] | None = None,
    captioner: ImageCaptioner | None = None,
    vlm_provider: str | None = None,
) -> list[Document]:
    """Universal document loader for files, directories, URLs, or lists of sources."""
    if source is None:
        return []

    if isinstance(source, list):
        documents = []
        for item in source:
            documents.extend(load_documents(item, captioner=captioner, vlm_provider=vlm_provider))
        return documents

    return IngestionRouter(captioner=captioner, vlm_provider=vlm_provider).load(source)
