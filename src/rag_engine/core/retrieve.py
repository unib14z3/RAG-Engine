from llama_index.core import VectorStoreIndex


def create_retriever(index: VectorStoreIndex, top_k: int = 5):
    if index is None:
        raise ValueError(
            "VectorStoreIndex is None. Run build_index() or ingest_documents() first."
        )

    return index.as_retriever(
        similarity_top_k=top_k
    )