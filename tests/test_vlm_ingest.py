import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from rag_engine.ingest import ImageIngester, PDFIngester, IngestionRouter, load_documents
from rag_engine.vlm import GeminiVLMClient, ImageCaptioner, LMStudioVLMClient


class TestVLMIngest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create a sample test image using PIL
        self.test_img_path = self.tmp_path / "test_diagram.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(self.test_img_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("rag_engine.vlm.gemini_client.genai.Client")
    def test_gemini_vlm_client_generate_caption(self, mock_genai_cls):
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Gemini caption of blue test diagram."
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiVLMClient(api_key="test-key", model_name="gemini-3.6-flash")
        caption = client.generate_caption(self.test_img_path)

        self.assertEqual(caption, "Gemini caption of blue test diagram.")
        mock_client.models.generate_content.assert_called_once()

    @patch("rag_engine.vlm.client.OpenAI")
    def test_vlm_client_generate_caption_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A test blue square diagram."
        mock_response.choices[0].message.reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response

        client = LMStudioVLMClient(base_url="http://localhost:2000/v1")
        caption = client.generate_caption(self.test_img_path)

        self.assertEqual(caption, "A test blue square diagram.")
        mock_client.chat.completions.create.assert_called_once()

    @patch("rag_engine.vlm.client.OpenAI")
    def test_vlm_client_server_unavailable(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        client = LMStudioVLMClient(base_url="http://localhost:2000/v1")
        caption = client.generate_caption(self.test_img_path)

        self.assertIsNone(caption)

    def test_image_captioner_provider_auto(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            captioner = ImageCaptioner(provider="auto")
            self.assertIsInstance(captioner.client, GeminiVLMClient)

        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=True):
            captioner = ImageCaptioner(provider="lmstudio")
            self.assertIsInstance(captioner.client, LMStudioVLMClient)

    def test_image_ingester_can_handle(self):
        captioner = MagicMock()
        ingester = ImageIngester(captioner=captioner)

        img1 = self.tmp_path / "diagram.png"
        img1.touch()
        img2 = self.tmp_path / "photo.jpg"
        img2.touch()
        img3 = self.tmp_path / "chart.webp"
        img3.touch()
        pdf_file = self.tmp_path / "document.pdf"
        pdf_file.touch()
        txt_file = self.tmp_path / "notes.txt"
        txt_file.touch()

        self.assertTrue(ingester.can_handle(img1))
        self.assertTrue(ingester.can_handle(img2))
        self.assertTrue(ingester.can_handle(img3))
        self.assertFalse(ingester.can_handle(pdf_file))
        self.assertFalse(ingester.can_handle(txt_file))

    def test_image_ingester_load(self):
        mock_captioner = MagicMock()
        mock_captioner.caption_image.return_value = "Detailed flowchart describing RAG architecture."

        ingester = ImageIngester(captioner=mock_captioner)
        docs = ingester.load(self.test_img_path)

        self.assertEqual(len(docs), 1)
        self.assertIn("Detailed flowchart describing RAG architecture", docs[0].text)
        self.assertEqual(docs[0].metadata["source_type"], "image")
        self.assertEqual(docs[0].metadata["content_type"], "image_caption")
        self.assertTrue(docs[0].metadata["is_visual_element"])

    @patch("pymupdf.open")
    def test_pdf_ingester_with_images(self, mock_pymupdf_open):
        mock_pdf = MagicMock()
        mock_pymupdf_open.return_value = mock_pdf

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text from PDF page."
        mock_page.get_images.return_value = [(100, 0, 100, 100, 8, "DeviceRGB", "", "img1", "FlateDecode")]
        mock_pdf.pages.return_value = [mock_page]

        with open(self.test_img_path, "rb") as f:
            img_bytes = f.read()

        mock_pdf.extract_image.return_value = {
            "image": img_bytes,
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        mock_captioner = MagicMock()
        mock_captioner.caption_image.return_value = "Chart showing performance metrics."

        ingester = PDFIngester(captioner=mock_captioner)
        pdf_file = self.tmp_path / "test_report.pdf"
        pdf_file.touch()

        docs = ingester.load(pdf_file)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["content_type"], "text")
        self.assertEqual(docs[1].metadata["content_type"], "image_caption")
        self.assertIn("Chart showing performance metrics", docs[1].text)

    def test_router_includes_image_ingester(self):
        mock_captioner = MagicMock()
        mock_captioner.caption_image.return_value = "System component diagram."

        docs = load_documents(self.test_img_path, captioner=mock_captioner)
        self.assertEqual(len(docs), 1)
        self.assertIn("System component diagram", docs[0].text)


if __name__ == "__main__":
    unittest.main()
