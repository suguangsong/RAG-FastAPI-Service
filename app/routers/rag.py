"""RAG 检索和问答路由"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from app.models import SearchRequest, SearchResponse, ChatRequest, ChatResponse
from app.services.retrieval_service import RetrievalService
from app.services.rerank_service import RerankService
from app.services.llm_service import LLMService
from app.config import settings
import json

router = APIRouter(prefix="/v1/rag", tags=["RAG"])
retrieval_service = RetrievalService()
rerank_service = RerankService()
llm_service = LLMService()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """向量检索"""
    try:
        results = await retrieval_service.hybrid_search(
            query=request.query,
            collection_name=request.collection_name or settings.qdrant_collection_name,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            use_hybrid=request.use_hybrid
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 问答"""
    try:
        # 检索相关文档
        search_results = await retrieval_service.hybrid_search(
            query=request.query,
            collection_name=request.collection_name or settings.qdrant_collection_name,
            top_k=request.top_k * 2 if request.use_rerank else request.top_k,
            use_hybrid=True
        )
        
        # 重排序
        if request.use_rerank and search_results:
            search_results = await rerank_service.rerank(
                query=request.query,
                documents=search_results,
                top_k=request.top_k
            )
        else:
            search_results = search_results[:request.top_k]
        
        # 流式输出
        if request.stream:
            return StreamingResponse(
                _stream_chat(request.query, search_results, request),
                media_type="text/event-stream"
            )
        
        # 非流式输出
        result = await llm_service.generate(
            query=request.query,
            context=search_results,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=search_results,
            usage=result.get("usage")
        )
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


async def _stream_chat(query: str, context: list, request: ChatRequest):
    """流式输出生成器"""
    try:
        async for chunk in llm_service.generate(
            query=query,
            context=context,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        ):
            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流式生成失败: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

