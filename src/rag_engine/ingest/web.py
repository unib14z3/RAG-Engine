from html.parser import HTMLParser
from pathlib import Path
import re
import urllib.request

from llama_index.core import Document

from .base import BaseIngester


class _HTMLTextExtractor(HTMLParser):
    """Simple, dependency-free HTML parser to extract clean text and title."""

    def __init__(self):
        super().__init__()
        self.result = []
        self.title = ""
        self._in_title = False
        self._skip = False

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript", "header", "footer", "nav"):
            self._skip = True
        elif tag_lower == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript", "header", "footer", "nav"):
            self._skip = False
        elif tag_lower == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self) -> str:
        raw_text = " ".join(self.result)
        return re.sub(r"\s+", " ", raw_text).strip()


def fetch_url_content(url: str, timeout: int = 10) -> tuple[str, str]:
    """Fetch web content from a URL and return (title, text_content)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAG-Engine/1.0"
    }
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=")[-1].split(";")[0].strip()

        html_bytes = response.read()
        try:
            html = html_bytes.decode(charset)
        except (UnicodeDecodeError, LookupError):
            html = html_bytes.decode("utf-8", errors="ignore")

    parser = _HTMLTextExtractor()
    parser.feed(html)
    title = parser.title.strip() or url
    text = parser.get_text()
    return title, text


def ingest_web(urls: str | list[str], timeout: int = 10) -> list[Document]:
    """Fetch web content from URL or list of URLs and return a list of Documents.

    Args:
        urls: A URL string or a list of URL strings.
        timeout: Request timeout in seconds.

    Returns:
        A list of llama_index Document objects containing extracted text and metadata.
    """
    if isinstance(urls, str):
        urls = [urls]

    documents = []
    for url in urls:
        url_str = str(url).strip()
        if not url_str.startswith(("http://", "https://")):
            url_str = "https://" + url_str

        print(f"Fetching web content: {url_str}")
        try:
            title, text = fetch_url_content(url_str, timeout=timeout)
            if not text:
                print(f"Warning: No readable text found at {url_str}")
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "url": url_str,
                        "title": title,
                        "source_type": "web",
                    },
                )
            )
        except Exception as e:
            print(f"Error fetching URL {url_str}: {e}")

    return documents


class WebIngester(BaseIngester):
    """Ingester for web pages specified by URL."""

    def can_handle(self, source: str | Path) -> bool:
        src = str(source).strip()
        return src.startswith(("http://", "https://"))

    def load(self, source: str | Path) -> list[Document]:
        return ingest_web(str(source))
