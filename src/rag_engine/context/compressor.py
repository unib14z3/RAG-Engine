import re

from .scorer import WeightedContextItem


class ContextCompressor:
    """Compacts context by performing sentence-level extractive compression and budget truncation."""

    def __init__(self, min_sentence_score: float = 0.1, char_per_token: float = 4.0):
        self.min_sentence_score = min_sentence_score
        self.char_per_token = char_per_token

    def _split_sentences(self, text: str) -> list[str]:
        """Splits text into individual sentences while retaining punctuation."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_query(self, query: str) -> set[str]:
        """Extracts unique lowercased query terms (excluding trivial short words)."""
        words = re.findall(r"\b\w{2,}\b", query.lower())
        return set(words)

    def compress_text(self, text: str, query: str = "") -> str:
        """Extractively prunes sentences from text that lack overlap with query terms.

        If query is empty or short, returns clean text without sentence pruning.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 2 or not query:
            return text

        query_terms = self._tokenize_query(query)
        if not query_terms:
            return text

        selected_sentences: list[str] = []
        for i, sentence in enumerate(sentences):
            sent_words = set(re.findall(r"\b\w{2,}\b", sentence.lower()))
            if not sent_words:
                continue

            overlap = len(query_terms.intersection(sent_words))
            overlap_score = overlap / len(query_terms)

            # Keep sentence if it matches query terms, or if it's the opening sentence
            if overlap_score >= self.min_sentence_score or i == 0:
                selected_sentences.append(sentence)

        # Fallback to original text if pruning removed everything
        if not selected_sentences:
            return text

        return " ".join(selected_sentences)

    def fit_to_budget(
        self,
        items: list[WeightedContextItem],
        max_tokens: int = 1024,
        query: str = "",
    ) -> list[WeightedContextItem]:
        
        max_chars = int(max_tokens * self.char_per_token)
        budget_items: list[WeightedContextItem] = []
        current_chars = 0

        for item in items:
            compacted_text = self.compress_text(item.text, query=query)
            item_chars = len(compacted_text)

            if current_chars + item_chars <= max_chars:
                item.text = compacted_text
                budget_items.append(item)
                current_chars += item_chars
            else:
                # Truncate final item to remaining character budget if room permits
                remaining_chars = max_chars - current_chars
                if remaining_chars >= 100:
                    truncated_text = compacted_text[:remaining_chars].rsplit(" ", 1)[0] + "..."
                    item.text = truncated_text
                    budget_items.append(item)
                break

        return budget_items
