"""自定义 Embeddings 实现"""
from typing import List
import httpx
from langchain_core.embeddings import Embeddings
from loguru import logger
from app.config import settings


class CustomJinaEmbeddings(Embeddings):
    """自定义 Jina Embeddings"""

    def __init__(self, model: str, jina_api_key: str):
        self.model = model
        self.jina_api_key = jina_api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        import asyncio
        return asyncio.run(self.aembed_documents(texts))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                        "task": "retrieval"
                    },
                    timeout=120.0
                )
                if response.status_code == 200:
                    result = response.json()
                    if "data" in result and len(result["data"]) > 0:
                        return [item["embedding"] for item in result["data"]]
                    else:
                        raise ValueError("Jina API 返回数据格式错误")
                else:
                    error_msg = response.text
                    raise ValueError(f"Jina API 错误: {response.status_code}, {error_msg}")
        except Exception as e:
            logger.error(f"Jina Embedding 生成失败: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        import asyncio
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询"""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]


class CustomDashScopeEmbeddings(Embeddings):
    """自定义 DashScope Embeddings"""

    def __init__(self, model: str, dashscope_api_key: str):
        self.model = model
        self.dashscope_api_key = dashscope_api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        import asyncio
        return asyncio.run(self.aembed_documents(texts))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档"""
        try:
            import dashscope
            dashscope.api_key = self.dashscope_api_key
            embeddings = []
            for text in texts:
                response = dashscope.Embedding.call(
                    model=self.model,
                    input=text
                )
                if response.status_code == 200:
                    embeddings.append(response.output['embeddings'][0]['embedding'])
                else:
                    raise ValueError(f"DashScope API 错误: {response.message}")
            return embeddings
        except Exception as e:
            logger.error(f"DashScope Embedding 生成失败: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        import asyncio
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询"""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
