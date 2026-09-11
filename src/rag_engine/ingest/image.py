import logging
from pathlib import Path

from llama_index.core import Document

from ..vlm import ImageCaptioner
from .base import BaseIngester

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageIngester(BaseIngester):
    """Ingester for indexing standalone image files via VLM image captioning."""

    def __init__(self, captioner: ImageCaptioner | None = None):
        self.captioner = captioner or ImageCaptioner()

    def can_handle(self, source: str | Path) -> bool:
        p = Path(source)
        return p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def load(self, source: str | Path) -> list[Document]:
        img_path = Path(source).expanduser().resolve()

        if not img_path.is_file():
            raise FileNotFoundError(f"Image source path does not exist: {img_path}")

        caption = self.captioner.caption_image(img_path)
        if not caption:
            logger.warning("No caption generated for image %s", img_path.name)
            return []

        return [
            Document(
                text=f"[Standalone Image: {img_path.name}]: {caption}",
                metadata={
                    "file_name": img_path.name,
                    "file_path": str(img_path),
                    "source_type": "image",
                    "content_type": "image_caption",
                    "is_visual_element": True,
                },
            )
        ]
