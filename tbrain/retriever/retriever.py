from tbrain.retriever.bm25_retriever import BM25Retriever


class Retriever:
    @classmethod
    def get_retriever(cls, name: str):
        if name == "bm25":
            return BM25Retriever
        else:
            raise ValueError(f"Unsupported retriever: {name}")
