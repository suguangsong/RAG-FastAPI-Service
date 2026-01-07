"""Embedding 服务模块 - 使用 LangChain"""
from typing import List, Optional
import numpy as np
from loguru import logger
from app.config import settings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_community.embeddings import JinaEmbeddings, DashScopeEmbeddings, OllamaEmbeddings
except ImportError:
    # 如果 langchain_community 没有这些类，使用自定义实现
    from langchain_community.embeddings import OllamaEmbeddings
    # Jina 和 DashScope 可能需要自定义实现
    JinaEmbeddings = None
    DashScopeEmbeddings = None


class EmbeddingService:
    """Embedding 服务 - 基于 LangChain"""

    def __init__(self):
        self.model = settings.embedding_model
        self.model_name = settings.embedding_model_name
        self._embeddings: Optional[Embeddings] = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """初始化 LangChain Embeddings"""
        try:
            if self.model == "jina":
                if JinaEmbeddings is None:
                    # 使用自定义 Jina Embeddings
                    from app.services.custom_embeddings import CustomJinaEmbeddings
                    self._embeddings = CustomJinaEmbeddings(
                        model=self.model_name,
                        jina_api_key=settings.jina_api_key
                    )
                else:
                    self._embeddings = JinaEmbeddings(
                        model=self.model_name,
                        jina_api_key=settings.jina_api_key
                    )
            elif self.model == "openai":
                self._embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=settings.openai_api_key
                )
            elif self.model == "dashscope":
                if DashScopeEmbeddings is None:
                    # 使用自定义 DashScope Embeddings
                    from app.services.custom_embeddings import CustomDashScopeEmbeddings
                    self._embeddings = CustomDashScopeEmbeddings(
                        model=self.model_name,
                        dashscope_api_key=settings.dashscope_api_key
                    )
                else:
                    self._embeddings = DashScopeEmbeddings(
                        model=self.model_name,
                        dashscope_api_key=settings.dashscope_api_key
                    )
            elif self.model == "ollama":
                self._embeddings = OllamaEmbeddings(
                    model=settings.ollama_embedding_model,
                    base_url=settings.ollama_base_url
                )
            else:
                raise ValueError(f"不支持的 Embedding 模型: {self.model}")
            logger.info(f"Embedding 初始化成功: {self.model}/{self.model_name}")
        except Exception as e:
            logger.error(f"Embedding 初始化失败: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
        try:
            embeddings = await self._embeddings.aembed_documents(texts)
            return self._normalize_embeddings(embeddings)
        except Exception as e:
            logger.error(f"Embedding 生成失败: {str(e)}")
            raise

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """L2 归一化向量"""
        normalized = []
        for emb in embeddings:
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                normalized.append((emb_array / norm).tolist())
            else:
                normalized.append(emb)
        return normalized

    async def embed_query(self, query: str) -> List[float]:
        """生成查询向量"""
        try:
            embedding = await self._embeddings.aembed_query(query)
            emb_array = np.array(embedding)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                return (emb_array / norm).tolist()
            return embedding
        except Exception as e:
            logger.error(f"查询向量生成失败: {str(e)}")
            raise
