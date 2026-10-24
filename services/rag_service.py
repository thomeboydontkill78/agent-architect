from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from core.vector_store import VectorStoreService
from core.model import get_chat_model

class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.chat_model = get_chat_model()
        self.prompt_template = PromptTemplate.from_file(settings.RAG_PROMPT_PATH)
        self.chain = self.prompt_template | self.chat_model | StrOutputParser()

    def _format(self, docs) -> str:
        return "\n\n".join(
            f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1)
        )

    def rag_summarize(self, query: str) -> str:
        docs = self.vector_store.retrieve(query)
        return self.chain.invoke({
            "input": query,
            "context": self._format(docs),
        })
