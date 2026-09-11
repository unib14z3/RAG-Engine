from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .cleaner import ContextCleaner
from .compressor import ContextCompressor
from .formatter import ContextFormatter
from .scorer import ContextScorer, WeightedContextItem


@dataclass
class ProcessedContextResult:
    formatted_text: str
    items: list[WeightedContextItem]
    total_characters: int
    estimated_tokens: int

    def __str__(self) -> str:
        return self.formatted_text


class ContextProcessor:
    """Orchestrates the context processing pipeline: Clean -> Deduplicate -> Score -> Compact -> Format."""

    def __init__(
        self,
        cleaner: Optional[ContextCleaner] = None,
        scorer: Optional[ContextScorer] = None,
        compressor: Optional[ContextCompressor] = None,
        formatter: Optional[ContextFormatter] = None,
    ):
        self.cleaner = cleaner or ContextCleaner()
        self.scorer = scorer or ContextScorer()
        self.compressor = compressor or ContextCompressor()
        self.formatter = formatter or ContextFormatter()

    def process(
        self,
        rag_items: list[Union[Dict[str, Any], Any]],
        web_items: Optional[list[Union[Dict[str, Any], Any]]] = None,
        query: str = "",
        max_tokens: int = 1024,
        output_format: str = "xml",
    ) -> ProcessedContextResult:
        """Processes raw RAG and Web retrieval items into a clean, weighted, compacted context string."""

        def _normalize(item: Any) -> Dict[str, Any]:
            if isinstance(item, dict):
                return item
            text = getattr(item, "text", getattr(item, "get_content", lambda: str(item))())
            metadata = getattr(item, "metadata", {})
            score = getattr(item, "score", 1.0)
            node_id = getattr(item, "node_id", None) or getattr(item, "id_", None)
            return {"text": str(text), "metadata": metadata, "score": score, "id": node_id}

        norm_rag = [_normalize(it) for it in (rag_items or [])]
        norm_web = [_normalize(it) for it in (web_items or [])]

        # 1. Clean & Deduplicate
        dedup_rag = self.cleaner.deduplicate(norm_rag)
        dedup_web = self.cleaner.deduplicate(norm_web)

        # 2. Score & Assign Weights
        weighted_rag = self.scorer.score_items(dedup_rag, source_type="rag")
        weighted_web = self.scorer.score_items(dedup_web, source_type="web")

        all_weighted = weighted_rag + weighted_web
        all_weighted.sort(key=lambda x: x.composite_weight, reverse=True)

        # 3. Compact & Fit to Token Budget
        compacted_items = self.compressor.fit_to_budget(
            all_weighted, max_tokens=max_tokens, query=query
        )

        # 4. Format Output
        if output_format.lower() == "markdown":
            formatted_text = self.formatter.to_markdown(compacted_items)
        elif output_format.lower() == "json":
            formatted_text = self.formatter.to_json(compacted_items)
        else:
            formatted_text = self.formatter.to_xml(compacted_items)

        total_chars = sum(len(it.text) for it in compacted_items)
        est_tokens = int(total_chars / self.compressor.char_per_token)

        return ProcessedContextResult(
            formatted_text=formatted_text,
            items=compacted_items,
            total_characters=total_chars,
            estimated_tokens=est_tokens,
        )
