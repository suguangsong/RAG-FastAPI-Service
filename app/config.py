"""配置管理模块"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""

    # Qdrant 配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "default"

    # Embedding 模型配置
    embedding_model: str = "openai"
    embedding_model_name: str = "text-embedding-3-small"
    openai_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # LLM 配置
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    ollama_llm_model: str = "llama2"

    # Rerank 配置
    use_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # 检索配置
    default_top_k: int = 10
    hybrid_search_alpha: float = 0.7
    rerank_top_k: int = 3

    # 文档处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size: int = 52428800

    # 服务配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

