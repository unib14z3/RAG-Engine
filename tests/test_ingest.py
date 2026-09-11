import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_engine import RAGPipeline, build_index, ingest_web
from rag_engine.ingest import (
    PDFIngester,
    TextIngester,
    WebIngester,
    IngestionRouter,
    load_documents,
)


class TestIngestSubmodule(unittest.TestCase):

    def test_text_ingester(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_txt = Path(tmp_dir) / "sample.txt"
            sample_txt.write_text(
                "Hello World! This is a test file for RAG engine text ingestion.",
                encoding="utf-8",
            )

            ingester = TextIngester()
            self.assertTrue(ingester.can_handle(sample_txt))

            docs = ingester.load(sample_txt)
            self.assertEqual(len(docs), 1)
            self.assertIn("Hello World!", docs[0].text)
            self.assertEqual(docs[0].metadata["source_type"], "text")

    def test_web_ingester_can_handle(self):
        ingester = WebIngester()
        self.assertTrue(ingester.can_handle("https://example.com"))
        self.assertTrue(ingester.can_handle("http://test.org/page"))
        self.assertFalse(ingester.can_handle("/path/to/local/file.txt"))

    @patch("urllib.request.urlopen")
    def test_ingest_web_function(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "text/html; charset=utf-8"
        mock_response.read.return_value = (
            b"<html><head><title>Test Page</title></head>"
            b"<body><p>Web content example</p></body></html>"
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = WebIngester().load("https://example.com")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["url"], "https://example.com")
        self.assertEqual(docs[0].metadata["title"], "Test Page")
        self.assertIn("Web content example", docs[0].text)

    def test_router_directory_scan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            txt_file = tmp_path / "file1.txt"
            txt_file.write_text("Text file content", encoding="utf-8")

            md_file = tmp_path / "file2.md"
            md_file.write_text("Markdown file content", encoding="utf-8")

            docs = load_documents(tmp_path)
            self.assertEqual(len(docs), 2)


if __name__ == "__main__":
    unittest.main()
