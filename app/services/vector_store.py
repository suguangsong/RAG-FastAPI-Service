"""向量存储服务模块 - 使用 LangChain"""
from typing import List, Dict, Any, Optional
from loguru import logger
from app.config import settings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class VectorStore:
    """向量存储服务 - 基于 LangChain Qdrant"""

    def __init__(self, embeddings):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self.embeddings = embeddings
        self._vector_store: Optional[QdrantVectorStore] = None

    def _get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """获取或创建向量存储"""
        if self._vector_store is None or collection_name != self.collection_name:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings
            )
            self.collection_name = collection_name
        return self._vector_store

    async def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]] = None,
        metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """添加文档到向量库"""
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # 创建 LangChain Document 对象
            documents = [
                Document(
                    page_content=text,
                    metadata=metadata or {}
                )
                for text, metadata in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            # 使用 LangChain 添加文档（会自动生成向量）
            ids = await vector_store.aadd_documents(documents)
            logger.info(f"成功添加 {len(documents)} 个文档到集合: {collection_name}")
            return ids if isinstance(ids, list) else [str(id) for id in ids]
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # 构建查询
            query = None
            if filter_condition:
                # 构建 Qdrant Filter
                conditions = []
                for key, value in filter_condition.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                from qdrant_client.models import Filter as QdrantFilter
                query_filter = QdrantFilter(must=conditions)
            else:
                query_filter = None
            
            # 使用 LangChain 搜索（通过查询文本）
            # 注意：LangChain Qdrant 主要通过文本搜索，我们需要使用相似度搜索
            # 如果支持向量搜索，使用 asimilarity_search_with_score_by_vector
            # 否则使用 asimilarity_search_with_score
            try:
                # 尝试使用向量搜索
                results = await vector_store.asimilarity_search_with_score_by_vector(
                    embedding=query_vector,
                    k=top_k,
                    score_threshold=score_threshold,
                    filter=query_filter
                )
            except AttributeError:
                # 如果不支持向量搜索，使用文本搜索（需要先获取查询文本）
                # 这里我们需要回退到直接使用 Qdrant 客户端
                from qdrant_client.models import SearchRequest
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=query_filter
                )
                results = []
                for result in search_result:
                    doc = Document(
                        page_content=result.payload.get("content", ""),
                        metadata=result.payload
                    )
                    results.append((doc, result.score))
            
            # 转换为统一格式
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "chunk_id"}
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise

    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """根据文档 ID 删除所有相关向量"""
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # 使用 Qdrant 客户端直接删除
            filter_condition = Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            )
            
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            if scroll_result[0]:
                point_ids = [point.id for point in scroll_result[0]]
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids
                )
                deleted_count = len(point_ids)
                logger.info(f"删除文档 {doc_id} 的 {deleted_count} 个向量")
                return deleted_count
            return 0
        except Exception as e:
            logger.error(f"删除文档向量失败: {str(e)}")
            raise

    async def get_document_chunks(
        self,
        collection_name: str,
        doc_id: str = None
    ) -> List[Dict[str, Any]]:
        """获取文档的所有 chunks"""
        try:
            filter_condition = None
            if doc_id:
                filter_condition = Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            if scroll_result[0]:
                for point in scroll_result[0]:
                    chunks.append({
                        "chunk_id": point.payload.get("chunk_id", ""),
                        "content": point.payload.get("content", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k not in ["chunk_id", "content"]}
                    })
            
            return chunks
        except Exception as e:
            logger.error(f"获取文档 chunks 失败: {str(e)}")
            raise
