import logging
import os
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from ..core.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, VLM_TIMEOUT

logger = logging.getLogger(__name__)


class GeminiVLMClient:
    """Client for Google Gemini API for VLM image understanding using google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
        timeout: float = VLM_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.timeout = timeout
        self.client = None

        try:
            http_opts = types.HttpOptions(timeout=int(self.timeout * 1000))
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key, http_options=http_opts)
            else:
                self.client = genai.Client(http_options=http_opts)
        except Exception as e:
            logger.debug("GeminiVLMClient failed to initialize client: %s", e)
            self.client = None

    def is_available(self) -> bool:
        """Check if Gemini client is initialized."""
        return self.client is not None

    def generate_caption(
        self,
        image_path: str | Path,
        prompt: str = (
            "Examine this visual element extracted from a document. "
            "Transcribe any prominent text, labels, chart data, or diagram structure "
            "in 2-4 concise technical sentences."
        ),
    ) -> str | None:
        """Send image file to Gemini API and return text caption."""
        if not self.client:
            logger.warning("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            return None

        path = Path(image_path).resolve()
        if not path.is_file():
            logger.error("Image file does not exist: %s", path)
            return None

        try:
            image = Image.open(path)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, prompt],
            )
            caption = response.text
            return caption.strip() if caption else None

        except Exception as e:
            logger.error("Gemini VLM captioning error for %s: %s", path.name, e)
            return None
