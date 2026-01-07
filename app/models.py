"""数据模型定义"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """文档状态"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadRequest(BaseModel):
    """上传文档请求"""
    collection_name: Optional[str] = Field(default="default", description="知识库名称")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="自定义元数据")


class UploadResponse(BaseModel):
    """上传文档响应"""
    doc_id: str
    filename: str
    status: str
    chunks_count: int
    message: str


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    filename: str
    status: str
    chunks_count: int
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    page: int
    page_size: int
    documents: List[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """删除文档响应"""
    doc_id: str
    message: str
    deleted_chunks: int


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="查询文本")
    collection_name: Optional[str] = Field(default="default", description="知识库名称")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1, description="相似度阈值")
    use_hybrid: bool = Field(default=True, description="是否使用混合检索")


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """检索响应"""
    query: str
    results: List[SearchResult]
    total: int


class ChatRequest(BaseModel):
    """问答请求"""
    query: str = Field(..., description="用户问题")
    collection_name: Optional[str] = Field(default="default", description="知识库名称")
    stream: bool = Field(default=False, description="是否流式输出")
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="LLM 温度参数")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大生成 token 数")
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回的文档数量")
    use_rerank: bool = Field(default=True, description="是否使用重排序")


class UsageInfo(BaseModel):
    """Token 使用信息"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[SearchResult]
    usage: Optional[UsageInfo] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: Dict[str, Any]

