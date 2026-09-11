import logging
from pathlib import Path

import pymupdf
from llama_index.core import Document

from ..core.config import EXTRACTED_IMAGES_DIR
from ..vlm import ImageCaptioner
from .base import BaseIngester

logger = logging.getLogger(__name__)


class PDFIngester(BaseIngester):
    """Ingester for extracting text and embedded images from PDF documents using PyMuPDF and VLM captioning."""

    def __init__(self, captioner: ImageCaptioner | None = None):
        self.captioner = captioner or ImageCaptioner()

    def can_handle(self, source: str | Path) -> bool:
        p = Path(source)
        return p.is_file() and p.suffix.lower() == ".pdf"

    def load(self, source: str | Path) -> list[Document]:
        pdf_path = Path(source).expanduser().resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF source path does not exist: {pdf_path}")

        documents = []
        pdf = pymupdf.open(pdf_path)
        EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        for page_number, page in enumerate(pdf.pages(), start=1):
            # 1. Page Text Extraction
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "file_name": pdf_path.name,
                            "file_path": str(pdf_path),
                            "page_number": page_number,
                            "source_type": "pdf",
                            "content_type": "text",
                        },
                    )
                )

            # 2. Page Embedded Image Extraction & VLM Captioning
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list, start=1):
                try:
                    xref = img_info[0]
                    base_image = pdf.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue

                    img_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip small decorative icons or logos (< 60px or < 50 bytes)
                    if (width and width < 60) or (height and height < 60) or len(img_bytes) < 50:
                        continue

                    img_filename = f"{pdf_path.stem}_p{page_number}_img{img_index}.{ext}"
                    img_path = EXTRACTED_IMAGES_DIR / img_filename
                    img_path.write_bytes(img_bytes)

                    # Generate VLM Caption
                    caption = self.captioner.caption_image(img_path)
                    if caption:
                        documents.append(
                            Document(
                                text=f"[Visual Content / Diagram on Page {page_number}]: {caption}",
                                metadata={
                                    "file_name": pdf_path.name,
                                    "file_path": str(pdf_path),
                                    "page_number": page_number,
                                    "image_path": str(img_path),
                                    "source_type": "pdf_image",
                                    "content_type": "image_caption",
                                    "is_visual_element": True,
                                },
                            )
                        )
                except Exception as e:
                    logger.warning(
                        "Failed extracting/captioning image %d on page %d of %s: %s",
                        img_index,
                        page_number,
                        pdf_path.name,
                        e,
                    )

        pdf.close()
        return documents
