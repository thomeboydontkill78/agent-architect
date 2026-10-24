# core/vector_store.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # RAG 踩过的坑，带上

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config.settings import settings
from core.model import get_embed_model   # ← 唯一升级：embeddings 走 model 层

class VectorStoreService:
    def __init__(self):
        self.embeddings = get_embed_model()   # ← 不再自己 OpenAIEmbeddings(...)
        self.chroma = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        return self._reranker

    def retrieve_coarse(self, query: str, k: int = 5) -> list[Document]:
        return self.chroma.similarity_search(query=query, k=k)

    def _rerank(self, docs: list[Document], query: str, k: int = 3) -> list[Document]:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:k]]

    def retrieve(self, query: str, top_k: int = 5, rerank_top_n: int = 3) -> list[Document]:
        return self._rerank(self.retrieve_coarse(query, top_k), query, rerank_top_n)
