from config.settings import settings
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

def get_chat_model():
    return ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.CHAT_MODEL_NAME,
        streaming=True,
    )

def get_embed_model():
    return OpenAIEmbeddings(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.EMBEDDING_MODEL_NAME,
        check_embedding_ctx_length=False
    )