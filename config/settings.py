from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):

    DASHSCOPE_API_KEY: str = Field(..., min_length=10, description="DashScope API Key")
    CHAT_MODEL_NAME: str = Field(default="qwen-max")
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-v3")
    RERANKER_MODEL_NAME: str = Field(default="BAAI/bge-reranker-v2-m3")

    TEXT_CHUNK_SIZE: int = Field(default=500)
    TEXT_CHUNK_OVERLAP: int = Field(default=50)
    COARSE_TOP_K: int = Field(default=5)
    CHROMA_PERSIST_DIR: str = Field(default="data/chroma_db")
    CHROMA_COLLECTION_NAME: str = Field(default="agent_kb")

    EXTERNAL_DATA_PATH: str = Field(default="data/external_data.csv")
    SYSTEM_PROMPT_PATH: str = Field(default="prompts/system.txt")
    REPORT_PROMPT_PATH: str = Field(default="prompts/report.txt")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()