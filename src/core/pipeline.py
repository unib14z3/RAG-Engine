from .config import TOP_K
from .ingest import build_index
from .retrieve import create_retriever


class RAGPipeline:

    def __init__(self):
        self.index = build_index()
        self.retriever = create_retriever(
            self.index,
            TOP_K,
        )

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)