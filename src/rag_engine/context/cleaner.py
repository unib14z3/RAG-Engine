import hashlib
import re
from typing import TypeVar

T = TypeVar("T")


class ContextCleaner:
    """Cleans raw context strings and eliminates duplicate or near-duplicate content chunks."""

    def __init__(self, jaccard_threshold: float = 0.75, ngram_size: int = 3):
        self.jaccard_threshold = jaccard_threshold
        self.ngram_size = ngram_size

    def clean_text(self, text: str) -> str:
        """Strips HTML tags, redundant markdown syntax, line breaks, and excess whitespace."""
        if not text:
            return ""

        # Remove HTML tags if present
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Replace non-breaking spaces and non-printable chars
        text = text.replace("\xa0", " ")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Collapse multiple spaces, tabs, and duplicate line breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,.!?:;])", r"\1", text)
        text = re.sub(r"\n\s*\n+", "\n", text)

        return text.strip()

    def _get_ngrams(self, text: str) -> set[str]:
        words = text.lower().split()
        if not words:
            return set()
        unigrams = set(words)
        bigrams = {
            " ".join(words[i : i + 2])
            for i in range(len(words) - 1)
        }
        return unigrams.union(bigrams)

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculates n-gram Jaccard similarity between two texts."""
        ngrams1 = self._get_ngrams(text1)
        ngrams2 = self._get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))

        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self, items: list[T], text_key: str = "text"
    ) -> list[T]:
        """Performs exact hash deduplication followed by n-gram Jaccard near-deduplication."""
        unique_items: list[T] = []
        seen_hashes: set[str] = set()

        for item in items:
            raw_text = (
                item.get(text_key, "")
                if isinstance(item, dict)
                else getattr(item, text_key, str(item))
            )
            cleaned = self.clean_text(raw_text)

            if not cleaned:
                continue

            # Exact hash check
            text_hash = hashlib.md5(cleaned.lower().encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                continue

            # Near-deduplication check against already accepted unique items
            is_near_dup = False
            for existing in unique_items:
                existing_text = (
                    existing.get(text_key, "")
                    if isinstance(existing, dict)
                    else getattr(existing, text_key, str(existing))
                )
                if (
                    self.jaccard_similarity(cleaned, existing_text)
                    >= self.jaccard_threshold
                ):
                    is_near_dup = True
                    break

            if not is_near_dup:
                seen_hashes.add(text_hash)
                unique_items.append(item)

        return unique_items
