import json

from .scorer import WeightedContextItem


class ContextFormatter:
    """Formats processed, weighted context items into structured LLM prompt blocks."""

    @staticmethod
    def to_xml(items: list[WeightedContextItem]) -> str:
        """Formats context items into structured XML blocks with metadata and weights."""
        if not items:
            return "<retrieved_context />"

        xml_parts = ["<retrieved_context>"]
        for item in items:
            source_attr = f'source="{item.source}"'
            weight_attr = f'weight="{item.composite_weight:.2f}"'
            id_attr = f'id="{item.id}"'
            
            # Format optional metadata attributes
            meta_attrs = ""
            if "url" in item.metadata:
                meta_attrs += f' url="{item.metadata["url"]}"'
            elif "file_name" in item.metadata:
                meta_attrs += f' file="{item.metadata["file_name"]}"'

            xml_parts.append(
                f'  <context_item {id_attr} {source_attr} {weight_attr}{meta_attrs}>\n'
                f"    {item.text}\n"
                f"  </context_item>"
            )

        xml_parts.append("</retrieved_context>")
        return "\n".join(xml_parts)

    @staticmethod
    def to_markdown(items: list[WeightedContextItem]) -> str:
        """Formats context items as a human-readable Markdown section."""
        if not items:
            return "*No context available.*"

        md_parts = ["### Retrieved Context"]
        for idx, item in enumerate(items, 1):
            source_label = "Local Document" if item.source == "rag" else "Web Search"
            md_parts.append(
                f"#### [{idx}] {source_label} (Weight: {item.composite_weight:.2f})\n"
                f"{item.text}\n"
            )

        return "\n".join(md_parts)

    @staticmethod
    def to_json(items: list[WeightedContextItem]) -> str:
        """Serializes items into a JSON string."""
        return json.dumps([item.to_dict() for item in items], indent=2)
