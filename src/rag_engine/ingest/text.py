from pathlib import Path

from llama_index.core import Document

from .base import BaseIngester

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".rst",
    ".log",
}


class TextIngester(BaseIngester):
    """Ingester for text-based files (.txt, .md, .csv, .json, etc.)."""

    def can_handle(self, source: str | Path) -> bool:
        p = Path(source)
        return p.is_file() and p.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS

    def load(self, source: str | Path) -> list[Document]:
        text_path = Path(source).expanduser().resolve()

        if not text_path.is_file():
            raise FileNotFoundError(f"Text file path does not exist: {text_path}")

        try:
            content = text_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = text_path.read_text(encoding="latin-1", errors="ignore").strip()

        if not content:
            return []

        return [
            Document(
                text=content,
                metadata={
                    "file_name": text_path.name,
                    "file_path": str(text_path),
                    "source_type": "text",
                },
            )
        ]
