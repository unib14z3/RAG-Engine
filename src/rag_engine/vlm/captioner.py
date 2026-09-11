import logging
import os
from pathlib import Path

from ..core.config import GEMINI_API_KEY, VLM_ENABLED, VLM_PROVIDER, VLM_TIMEOUT
from ..utils import VLMTimerStatus
from .client import LMStudioVLMClient
from .gemini_client import GeminiVLMClient

logger = logging.getLogger(__name__)

DEFAULT_CAPTION_PROMPT = (
    "Examine this visual element extracted from a document. "
    "Transcribe any prominent text, labels, chart data, or diagram structure "
    "in 2-4 concise technical sentences."
)


class ImageCaptioner:
    """High-level service for generating searchable captions from images using Gemini API or LM Studio VLM."""

    def __init__(
        self,
        vlm_client: GeminiVLMClient | LMStudioVLMClient | None = None,
        enabled: bool = VLM_ENABLED,
        provider: str = VLM_PROVIDER,
    ):
        self.enabled = enabled
        self.provider = (provider or VLM_PROVIDER).lower().strip()

        if self.provider in ("none", "off", "disabled", "false"):
            self.enabled = False
            self.client = None
            logger.info("VLM Image Captioning is disabled.")
            return

        if vlm_client is not None:
            self.client = vlm_client
        else:
            has_gemini_key = bool(GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"))
            if self.provider == "gemini" or (self.provider == "auto" and has_gemini_key):
                logger.info("Initializing ImageCaptioner with Gemini VLM client...")
                self.client = GeminiVLMClient()
            elif self.provider == "lmstudio" or (self.provider == "auto" and not has_gemini_key):
                logger.info("Initializing ImageCaptioner with LM Studio local VLM client...")
                self.client = LMStudioVLMClient()
            else:
                logger.info("Initializing default ImageCaptioner (LM Studio)...")
                self.client = LMStudioVLMClient()

    def caption_image(
        self,
        image_path: str | Path,
        prompt: str = DEFAULT_CAPTION_PROMPT,
    ) -> str | None:
        """Generate a text description for an image with live timer UI. Returns None if VLM fails."""
        if not self.enabled or not self.client:
            logger.info("VLM captioning is disabled by configuration.")
            return None

        path = Path(image_path).resolve()
        provider_name = (
            f"Gemini API ({getattr(self.client, 'model_name', 'gemini-3.6-flash')})"
            if isinstance(self.client, GeminiVLMClient)
            else f"LM Studio ({getattr(self.client, 'model_name', 'gemma-4-e4b')})"
        )

        with VLMTimerStatus(path.name, provider_name=provider_name, timeout=VLM_TIMEOUT):
            caption = self.client.generate_caption(path, prompt=prompt)

        if caption:
            logger.info("Generated caption for image %s (%d chars)", path.name, len(caption))
        else:
            logger.warning("Could not generate caption for image %s", path.name)
        return caption
