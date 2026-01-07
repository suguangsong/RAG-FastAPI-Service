"""检索服务模块 - 混合检索"""
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.bm25_service import BM25Service


class RetrievalService:
    """检索服务 - 支持混合检索"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.bm25_service = BM25Service()
        self.alpha = settings.hybrid_search_alpha

    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        if use_hybrid:
            return await self._hybrid_search(query, collection_name, top_k, score_threshold)
        else:
            return await self._vector_search(query, collection_name, top_k, score_threshold)

    async def _vector_search(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        """纯向量检索"""
        query_vector = await self.embedding_service.embed_query(query)
        results = await self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return results

    async def _hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        """混合检索：向量检索 + BM25"""
        # 向量检索
        query_vector = await self.embedding_service.embed_query(query)
        vector_results = await self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k * 2
        )
        
        # 获取所有文档用于 BM25 索引
        # 注意：这里简化处理，实际应该预先构建索引或使用更高效的方式
        # 为了性能，我们使用向量检索结果来构建 BM25 索引
        if vector_results:
            try:
                self.bm25_service.build_index(collection_name, vector_results)
            except Exception as e:
                logger.warning(f"构建 BM25 索引失败，仅使用向量检索: {str(e)}")
                return vector_results[:top_k]
        
        # BM25 检索
        bm25_results = self.bm25_service.search(collection_name, query, top_k=top_k * 2)
        
        # 合并结果
        combined_results = self._combine_results(
            vector_results,
            bm25_results,
            collection_name,
            top_k
        )
        
        # 应用阈值过滤
        if score_threshold:
            combined_results = [
                r for r in combined_results
                if r["score"] >= score_threshold
            ]
        
        return combined_results[:top_k]

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, float]],
        collection_name: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """合并向量检索和 BM25 检索结果"""
        # 创建向量结果映射
        vector_map = {r["chunk_id"]: r for r in vector_results}
        
        # 获取 BM25 对应的文档
        bm25_map = {}
        try:
            # 从 BM25 服务获取元数据
            if collection_name in self.bm25_service.metadata:
                doc_metadata = self.bm25_service.metadata[collection_name]
                for bm25_result in bm25_results:
                    doc_idx = bm25_result["index"]
                    if doc_idx < len(doc_metadata):
                        chunk_id = doc_metadata[doc_idx].get("chunk_id", "")
                        if chunk_id:
                            bm25_map[chunk_id] = bm25_result["score"]
        except Exception as e:
            logger.warning(f"获取 BM25 文档映射失败: {str(e)}")
        
        # 归一化得分
        vector_scores = [r["score"] for r in vector_results]
        bm25_scores = list(bm25_map.values()) if bm25_map else []
        
        normalized_vector_scores = self._normalize_scores(vector_scores)
        normalized_bm25_scores = self.bm25_service.normalize_scores(bm25_scores) if bm25_scores else []
        
        # 合并得分
        combined_map = {}
        for idx, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            vector_score = normalized_vector_scores[idx] if idx < len(normalized_vector_scores) else 0.0
            bm25_score = bm25_map.get(chunk_id, 0.0)
            if bm25_score > 0 and normalized_bm25_scores:
                bm25_idx = list(bm25_map.keys()).index(chunk_id)
                bm25_score = normalized_bm25_scores[bm25_idx] if bm25_idx < len(normalized_bm25_scores) else 0.0
            
            combined_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            combined_map[chunk_id] = {
                **result,
                "score": combined_score,
                "vector_score": vector_score,
                "bm25_score": bm25_score
            }
        
        # 排序并返回
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化得分到 [0, 1] 区间"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized

