"""重排序服务模块"""
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from loguru import logger
from app.config import settings


class RerankService:
    """重排序服务"""

    def __init__(self):
        self.model = None
        self.use_rerank = settings.use_rerank
        if self.use_rerank:
            self._load_model()

    def _load_model(self):
        """加载重排序模型"""
        try:
            self.model = CrossEncoder(settings.rerank_model)
            logger.info(f"重排序模型加载成功: {settings.rerank_model}")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {str(e)}")
            self.use_rerank = False

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """对检索结果进行重排序"""
        if not self.use_rerank or not self.model or not documents:
            return documents[:top_k] if top_k else documents
        
        top_k = top_k or settings.rerank_top_k
        
        try:
            pairs = [[query, doc["content"]] for doc in documents]
            scores = self.model.predict(pairs)
            
            reranked_docs = []
            for idx, score in enumerate(scores):
                doc = documents[idx].copy()
                doc["rerank_score"] = float(score)
                reranked_docs.append(doc)
            
            reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            return reranked_docs[:top_k]
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return documents[:top_k]

