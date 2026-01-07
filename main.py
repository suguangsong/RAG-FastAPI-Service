"""主应用入口"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
from app.config import settings
from app.routers import documents, rag

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG FastAPI Service",
    description="企业级检索增强生成 (RAG) 异步服务框架",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents.router)
app.include_router(rag.router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG FastAPI Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """健康检查"""
    try:
        from app.services.vector_store import VectorStore
        vector_store = VectorStore()
        vector_store.client.get_collections()
        return {"status": "healthy", "qdrant": "connected"}
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

