import base64
import logging
from pathlib import Path

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from ..core.config import VLM_BASE_URL, VLM_MODEL_NAME, VLM_TIMEOUT

logger = logging.getLogger(__name__)


class LMStudioVLMClient:
    """Client for local LM Studio Vision-Language Model server."""

    def __init__(
        self,
        base_url: str = VLM_BASE_URL,
        model_name: str = VLM_MODEL_NAME,
        timeout: float = VLM_TIMEOUT,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="lm-studio",
            timeout=self.timeout,
        )

    def is_available(self) -> bool:
        """Check if LM Studio VLM server is reachable."""
        try:
            # Simple models list call to verify server endpoint
            self.client.models.list()
            return True
        except Exception as e:
            logger.debug("VLM server check failed at %s: %s", self.base_url, e)
            return False

    def generate_caption(
        self,
        image_path: str | Path,
        prompt: str = (
            "Analyze this image from a document. "
            "Transcribe all visible text, describe diagrams, flowcharts, tables, "
            "or visual structures in 2-4 concise technical sentences."
        ),
        max_tokens: int = 300,
    ) -> str | None:
        """Send image file to LM Studio VLM and return text caption."""
        path = Path(image_path).resolve()
        if not path.is_file():
            logger.error("Image file does not exist: %s", path)
            return None

        # Determine MIME type
        ext = path.suffix.lower().lstrip(".")
        mime_type = "image/png" if ext == "png" else f"image/{ext}"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        try:
            with open(path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()

            caption = content if content else reasoning
            return caption if caption else None

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning("LM Studio VLM server unreachable/timeout at %s: %s", self.base_url, e)
            return None
        except APIError as e:
            logger.error("API error from LM Studio VLM: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error during VLM captioning for %s: %s", path.name, e)
            return None
