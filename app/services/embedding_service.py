"""Embedding 服务模块"""
from typing import List
import numpy as np
from loguru import logger
from app.config import settings


class EmbeddingService:
    """Embedding 服务"""

    def __init__(self):
        self.model = settings.embedding_model
        self.model_name = settings.embedding_model_name
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 Embedding 客户端"""
        if self.model == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.error(f"OpenAI 客户端初始化失败: {str(e)}")
                raise
        elif self.model == "dashscope":
            try:
                import dashscope
                dashscope.api_key = settings.dashscope_api_key
                self._client = dashscope
            except Exception as e:
                logger.error(f"DashScope 客户端初始化失败: {str(e)}")
                raise
        elif self.model == "ollama":
            self._client = None
        else:
            raise ValueError(f"不支持的 Embedding 模型: {self.model}")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
        if self.model == "openai":
            return await self._embed_openai(texts)
        elif self.model == "dashscope":
            return await self._embed_dashscope(texts)
        elif self.model == "ollama":
            return await self._embed_ollama(texts)
        else:
            raise ValueError(f"不支持的 Embedding 模型: {self.model}")

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI 生成向量"""
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return self._normalize_embeddings(embeddings)
        except Exception as e:
            logger.error(f"OpenAI Embedding 生成失败: {str(e)}")
            raise

    async def _embed_dashscope(self, texts: List[str]) -> List[List[float]]:
        """使用 DashScope 生成向量"""
        try:
            import dashscope
            embeddings = []
            for text in texts:
                response = dashscope.Embedding.call(
                    model=self.model_name,
                    input=text
                )
                if response.status_code == 200:
                    embeddings.append(response.output['embeddings'][0]['embedding'])
                else:
                    raise ValueError(f"DashScope API 错误: {response.message}")
            return self._normalize_embeddings(embeddings)
        except Exception as e:
            logger.error(f"DashScope Embedding 生成失败: {str(e)}")
            raise

    async def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """使用 Ollama 生成向量"""
        try:
            import httpx
            embeddings = []
            async with httpx.AsyncClient() as client:
                for text in texts:
                    response = await client.post(
                        f"{settings.ollama_base_url}/api/embeddings",
                        json={
                            "model": settings.ollama_embedding_model,
                            "prompt": text
                        },
                        timeout=60.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        embeddings.append(result.get("embedding", []))
                    else:
                        raise ValueError(f"Ollama API 错误: {response.status_code}")
            return self._normalize_embeddings(embeddings)
        except Exception as e:
            logger.error(f"Ollama Embedding 生成失败: {str(e)}")
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
        embeddings = await self.embed_texts([query])
        return embeddings[0]

