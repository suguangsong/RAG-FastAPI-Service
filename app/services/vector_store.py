"""向量存储服务模块"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import uuid
from loguru import logger
from app.config import settings


class VectorStore:
    """向量存储服务"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name

    def ensure_collection(self, collection_name: str, vector_size: int = 1536):
        """确保集合存在"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建集合: {collection_name}")
        except Exception as e:
            logger.error(f"创建集合失败: {collection_name}, 错误: {str(e)}")
            raise

    async def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """添加文档到向量库"""
        self.ensure_collection(collection_name, len(embeddings[0]) if embeddings else 1536)
        
        points = []
        chunk_ids = []
        
        for idx, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            point = PointStruct(
                id=hash(chunk_id) % (2**63),
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "content": text,
                    **metadata
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"成功添加 {len(points)} 个文档到集合: {collection_name}")
            return chunk_ids
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
            search_filter = None
            if filter_condition:
                conditions = []
                for key, value in filter_condition.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                search_filter = Filter(must=conditions)
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            results = []
            for result in search_result:
                results.append({
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "content": result.payload.get("content", ""),
                    "score": float(result.score),
                    "metadata": {k: v for k, v in result.payload.items() if k not in ["chunk_id", "content"]}
                })
            
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise

    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """根据文档 ID 删除所有相关向量"""
        try:
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

