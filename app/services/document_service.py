"""文档服务模块"""
from typing import List, Dict, Any
import uuid
from datetime import datetime
from loguru import logger
from app.config import settings
from app.utils.document_parser import DocumentParser
from app.utils.text_splitter import RecursiveCharacterTextSplitter
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class DocumentService:
    """文档服务"""

    def __init__(self):
        self.parser = DocumentParser()
        self.splitter = RecursiveCharacterTextSplitter()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.documents: Dict[str, Dict[str, Any]] = {}

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        collection_name: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """上传并处理文档"""
        doc_id = str(uuid.uuid4())
        
        try:
            # 解析文档
            text = self.parser.parse(file_content, filename)
            
            # 创建文档记录
            doc_info = {
                "doc_id": doc_id,
                "filename": filename,
                "status": "processing",
                "chunks_count": 0,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }
            self.documents[doc_id] = doc_info
            
            # 文本切片
            chunks = self.splitter.create_chunks(
                text=text,
                doc_id=doc_id,
                filename=filename,
                metadata=metadata or {}
            )
            
            # 生成向量
            texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # 存储到向量库
            metadatas = [
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "filename": chunk["filename"],
                    **chunk["metadata"]
                }
                for chunk in chunks
            ]
            
            chunk_ids = await self.vector_store.add_documents(
                collection_name=collection_name,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            # 更新文档状态
            doc_info["status"] = "completed"
            doc_info["chunks_count"] = len(chunks)
            
            return {
                "doc_id": doc_id,
                "filename": filename,
                "status": "completed",
                "chunks_count": len(chunks),
                "message": "文档处理完成"
            }
        except Exception as e:
            logger.error(f"文档处理失败: {filename}, 错误: {str(e)}")
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = "failed"
            raise

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """获取文档信息"""
        if doc_id not in self.documents:
            raise ValueError(f"文档不存在: {doc_id}")
        return self.documents[doc_id]

    def list_documents(
        self,
        collection_name: str = None,
        status: str = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """列出文档"""
        docs = list(self.documents.values())
        
        if collection_name:
            docs = [d for d in docs if d.get("collection_name") == collection_name]
        
        if status:
            docs = [d for d in docs if d.get("status") == status]
        
        total = len(docs)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_docs = docs[start:end]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": paginated_docs
        }

    async def delete_document(self, doc_id: str, collection_name: str) -> Dict[str, Any]:
        """删除文档"""
        if doc_id not in self.documents:
            raise ValueError(f"文档不存在: {doc_id}")
        
        try:
            deleted_count = await self.vector_store.delete_by_doc_id(collection_name, doc_id)
            del self.documents[doc_id]
            
            return {
                "doc_id": doc_id,
                "message": "文档及向量索引已删除",
                "deleted_chunks": deleted_count
            }
        except Exception as e:
            logger.error(f"删除文档失败: {doc_id}, 错误: {str(e)}")
            raise

