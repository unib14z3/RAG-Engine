import unittest
from rag_engine.context import (
    ContextCleaner,
    ContextCompressor,
    ContextFormatter,
    ContextProcessor,
    ContextScorer,
    WeightedContextItem,
)


class TestContextPipeline(unittest.TestCase):

    def test_cleaner_html_and_deduplication(self):
        cleaner = ContextCleaner(jaccard_threshold=0.7)

        raw_html = "<p>Hello <b>World</b>! Check <a href='https://example.com'>this link</a>.</p>"
        cleaned = cleaner.clean_text(raw_html)
        self.assertNotIn("<p>", cleaned)
        self.assertIn("Hello World!", cleaned)

        items = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "The quick brown fox jumps over the lazy dog."},  # Exact duplicate
            {"text": "The quick brown fox jumps over a lazy dog."},  # Near duplicate
            {"text": "Artificial intelligence and machine learning are revolutionizing tech."},
        ]

        deduped = cleaner.deduplicate(items)
        self.assertEqual(len(deduped), 2)
        self.assertIn("Artificial intelligence", deduped[1]["text"])

    def test_scorer_weight_assignment(self):
        scorer = ContextScorer(rag_default_weight=0.95, web_default_weight=0.60)

        raw_rag = [{"text": "Internal document content.", "score": 0.9}]
        raw_web_gov = [
            {
                "text": "Government report summary.",
                "score": 0.8,
                "metadata": {"url": "https://data.gov/report"},
            }
        ]
        raw_web_blog = [
            {
                "text": "Random opinion post.",
                "score": 0.5,
                "metadata": {"url": "https://randomblog.com/post"},
            }
        ]

        scored_rag = scorer.score_items(raw_rag, source_type="rag")
        scored_gov = scorer.score_items(raw_web_gov, source_type="web")
        scored_blog = scorer.score_items(raw_web_blog, source_type="web")

        self.assertEqual(scored_rag[0].source_weight, 0.95)
        self.assertEqual(scored_gov[0].source_weight, 0.90)
        self.assertEqual(scored_blog[0].source_weight, 0.60)
        self.assertGreater(scored_rag[0].composite_weight, scored_blog[0].composite_weight)

    def test_compressor_sentence_pruning_and_budget(self):
        compressor = ContextCompressor(min_sentence_score=0.1)

        text = (
            "SIH is the Smart India Hackathon organized annually. "
            "Python is a popular programming language. "
            "Smart India Hackathon encourages student innovation across the nation."
        )
        compressed = compressor.compress_text(text, query="Smart India Hackathon SIH")
        self.assertIn("Smart India Hackathon", compressed)
        self.assertNotIn("Python is a popular", compressed)

        items = [
            WeightedContextItem(
                id="1",
                text="First sentence. Second sentence.",
                source="rag",
                composite_weight=0.9,
            ),
            WeightedContextItem(
                id="2",
                text="Third sentence. Fourth sentence.",
                source="web",
                composite_weight=0.7,
            ),
        ]

        budget_items = compressor.fit_to_budget(items, max_tokens=10)
        self.assertLessEqual(len(budget_items), 2)

    def test_formatter_xml_markdown_json(self):
        items = [
            WeightedContextItem(
                id="rag_1",
                text="Local context snippet.",
                source="rag",
                composite_weight=0.92,
                metadata={"file_name": "policy.pdf"},
            ),
            WeightedContextItem(
                id="web_1",
                text="Web search result snippet.",
                source="web",
                composite_weight=0.75,
                metadata={"url": "https://example.org/info"},
            ),
        ]

        xml_out = ContextFormatter.to_xml(items)
        self.assertIn("<retrieved_context>", xml_out)
        self.assertIn('id="rag_1"', xml_out)
        self.assertIn('weight="0.92"', xml_out)
        self.assertIn('source="rag"', xml_out)

        md_out = ContextFormatter.to_markdown(items)
        self.assertIn("### Retrieved Context", md_out)
        self.assertIn("Weight: 0.92", md_out)

        json_out = ContextFormatter.to_json(items)
        self.assertIn('"composite_weight": 0.92', json_out)

    def test_context_processor_full_pipeline(self):
        processor = ContextProcessor()

        rag_items = [
            {"text": "<p>Internal guidelines for SIH 2026 hackathon submission.</p>", "score": 0.95}
        ]
        web_items = [
            {
                "text": "Official website updates for SIH 2026 guidelines.",
                "score": 0.85,
                "metadata": {"url": "https://sih.gov.in"},
            }
        ]

        result = processor.process(
            rag_items=rag_items,
            web_items=web_items,
            query="SIH 2026 guidelines",
            max_tokens=500,
            output_format="xml",
        )

        self.assertGreater(result.estimated_tokens, 0)
        self.assertEqual(len(result.items), 2)
        self.assertIn("<retrieved_context>", result.formatted_text)
        self.assertIn('source="rag"', result.formatted_text)
        self.assertIn('source="web"', result.formatted_text)


if __name__ == "__main__":
    unittest.main()
