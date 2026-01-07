"""文档管理路由"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import Optional
from loguru import logger
from app.models import (
    UploadResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    ErrorResponse
)
from app.services.document_service import DocumentService
from app.config import settings

router = APIRouter(prefix="/v1", tags=["文档管理"])
document_service = DocumentService()


@router.post("/ingest/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
    metadata: Optional[str] = Form(default=None)
):
    """上传文档"""
    try:
        # 检查文件大小
        file_content = await file.read()
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制: {settings.max_file_size} bytes"
            )
        
        # 解析 metadata
        import json
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Metadata JSON 解析失败: {metadata}")
        
        # 处理文档
        result = await document_service.upload_document(
            file_content=file_content,
            filename=file.filename,
            collection_name=collection_name,
            metadata=doc_metadata
        )
        
        return UploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    collection_name: Optional[str] = Query(None, description="知识库名称"),
    status: Optional[str] = Query(None, description="文档状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """查询文档列表"""
    try:
        result = document_service.list_documents(
            collection_name=collection_name,
            status=status,
            page=page,
            page_size=page_size
        )
        return DocumentListResponse(**result)
    except Exception as e:
        logger.error(f"查询文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    doc_id: str,
    collection_name: str = Query("default", description="知识库名称")
):
    """删除文档"""
    try:
        result = await document_service.delete_document(doc_id, collection_name)
        return DeleteDocumentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

