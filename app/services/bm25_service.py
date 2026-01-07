"""BM25 关键词检索服务"""
from typing import List, Dict
import jieba
from rank_bm25 import BM25Okapi
from loguru import logger


class BM25Service:
    """BM25 关键词检索服务"""

    def __init__(self):
        self.indexes: Dict[str, BM25Okapi] = {}
        self.documents: Dict[str, List[str]] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}

    def build_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        """构建 BM25 索引"""
        tokenized_docs = []
        doc_texts = []
        doc_metadata = []
        
        for doc in documents:
            text = doc.get("content", "")
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            doc_texts.append(text)
            doc_metadata.append({
                "chunk_id": doc.get("chunk_id", ""),
                "content": text
            })
        
        if tokenized_docs:
            bm25 = BM25Okapi(tokenized_docs)
            self.indexes[collection_name] = bm25
            self.documents[collection_name] = doc_texts
            self.metadata[collection_name] = doc_metadata
            logger.info(f"为集合 {collection_name} 构建 BM25 索引，文档数: {len(documents)}")

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return jieba.lcut(text)

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, float]]:
        """BM25 搜索"""
        if collection_name not in self.indexes:
            logger.warning(f"集合 {collection_name} 的 BM25 索引不存在")
            return []
        
        bm25 = self.indexes[collection_name]
        tokenized_query = self._tokenize(query)
        
        scores = bm25.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                results.append({
                    "index": idx,
                    "score": float(score)
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化得分到 [0, 1] 区间"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized

