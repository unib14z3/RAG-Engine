import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class WeightedContextItem:
    """Represents a context chunk enriched with source metadata and composite weight scores."""

    id: str
    text: str
    source: str  # "rag" or "web"
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_score: float = 1.0
    source_weight: float = 1.0
    recency_score: float = 1.0
    composite_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "base_score": round(self.base_score, 4),
            "source_weight": round(self.source_weight, 4),
            "recency_score": round(self.recency_score, 4),
            "composite_weight": round(self.composite_weight, 4),
        }


class ContextScorer:
    """Calculates multi-factor weight scores for retrieved RAG and Web context items."""

    def __init__(
        self,
        rag_default_weight: float = 0.95,
        web_default_weight: float = 0.70,
        high_authority_domains: Optional[list[str]] = None,
        w_sim: float = 0.50,
        w_src: float = 0.35,
        w_rec: float = 0.15,
    ):
        self.rag_default_weight = rag_default_weight
        self.web_default_weight = web_default_weight
        self.high_authority_domains = set(
            high_authority_domains or [".gov", ".edu", ".org", "wikipedia.org", "github.com"]
        )
        self.w_sim = w_sim
        self.w_src = w_src
        self.w_rec = w_rec

    def _determine_source_weight(self, source: str, metadata: Dict[str, Any]) -> float:
        if source == "rag":
            return self.rag_default_weight

        url = metadata.get("url", "") or metadata.get("domain", "")
        if url:
            for domain in self.high_authority_domains:
                if domain in url.lower():
                    return 0.90

        return self.web_default_weight

    def _calculate_recency_score(self, metadata: Dict[str, Any], half_life_days: float = 180.0) -> float:
        date_val = metadata.get("published_date") or metadata.get("date") or metadata.get("created_at")
        if not date_val:
            return 1.0

        try:
            if isinstance(date_val, (int, float)):
                doc_dt = datetime.fromtimestamp(date_val, tz=timezone.utc)
            elif isinstance(date_val, str):
                doc_dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
            elif isinstance(date_val, datetime):
                doc_dt = date_val
            else:
                return 1.0

            now = datetime.now(timezone.utc)
            days_old = max(0.0, (now - doc_dt).total_seconds() / 86400.0)
            decay_rate = math.log(2) / half_life_days
            return math.exp(-decay_rate * days_old)
        except Exception:
            return 1.0

    def score_items(
        self,
        raw_items: list[Dict[str, Any]],
        source_type: str = "rag",
    ) -> list[WeightedContextItem]:
        """Scores raw context dicts and returns WeightedContextItem instances."""
        weighted_items: list[WeightedContextItem] = []

        for idx, item in enumerate(raw_items):
            text = item.get("text", "").strip()
            if not text:
                continue

            item_id = item.get("id") or f"{source_type}_{idx+1}"
            metadata = item.get("metadata", {})
            base_score = float(item.get("score", 1.0))

            src_weight = self._determine_source_weight(source_type, metadata)
            recency = self._calculate_recency_score(metadata)

            # Composite weight calculation
            raw_composite = (
                self.w_sim * base_score
                + self.w_src * src_weight
                + self.w_rec * recency
            )
            composite_weight = min(1.0, max(0.0, raw_composite))

            weighted_items.append(
                WeightedContextItem(
                    id=str(item_id),
                    text=text,
                    source=source_type,
                    metadata=metadata,
                    base_score=base_score,
                    source_weight=src_weight,
                    recency_score=recency,
                    composite_weight=composite_weight,
                )
            )

        # Sort descending by composite weight
        weighted_items.sort(key=lambda x: x.composite_weight, reverse=True)
        return weighted_items
