# 🚀 RAG-FastAPI-Service

这是一个基于 FastAPI 构建的企业级检索增强生成 (RAG) 异步服务框架。它提供了完整的文档生命周期管理，支持文档切片、自动向量化、混合检索。

## 🛠️ 系统架构

### 架构概览

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│  Qdrant     │
│             │     │  Service     │     │  Vector DB  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├──────────────┐
                           │              │
                    ┌──────▼──────┐ ┌────▼──────┐
                    │  Embedding  │ │    LLM    │
                    │   Model     │ │  Service  │
                    └─────────────┘ └───────────┘
```

### 数据流程

1. **文档入库流程:**
   ```
   上传文件 → 文档解析 → 文本切片 → Embedding 向量化 → 存储到 Qdrant
   ```

2. **检索问答流程:**
   ```
   用户问题 → 向量检索 + 关键词检索 → 混合排序 → Rerank → LLM 生成回答
   ```

### 技术栈
- **后端框架:** Python 3.10+ / FastAPI (异步支持)
- **RAG 编排:** LangChain / LlamaIndex
- **向量引擎:** Qdrant (存储向量 + 元数据，支持 HNSW 索引)
- **Embedding 模型:** OpenAI text-embedding-3-small / DashScope text-embedding-v2 / Ollama (本地)
- **LLM 支持:** OpenAI GPT-4 / DashScope Qwen / Ollama (本地)
- **文档解析:** PyPDF2 / python-docx / 其他文本解析库
- **重排序模型:** Cross-Encoder (可选，用于提升检索精度)

## 📌 接口规范 (API Endpoints)

系统通过 HTTP 接口对外提供服务，默认文档地址：`http://localhost:8000/docs`

### 1. 知识库管理

#### 1.1 上传文档
- **接口:** `POST /v1/ingest/upload`
- **功能:** 上传文件（PDF/Docx/TXT），自动进行切片和向量化
- **请求格式:**
  ```json
  {
    "file": "<binary_file>",
    "collection_name": "default",  // 可选，指定知识库名称
    "metadata": {                   // 可选，自定义元数据
      "source": "manual_upload",
      "category": "technical"
    }
  }
  ```
- **响应格式:**
  ```json
  {
    "doc_id": "uuid-string",
    "filename": "example.pdf",
    "status": "processing",
    "chunks_count": 0,
    "message": "文档已接收，正在处理中"
  }
  ```
- **处理流程:** 接收文件 → 文档解析 → 自动切片 → 生成 Embedding → 存入 Qdrant

#### 1.2 查询文档列表
- **接口:** `GET /v1/documents`
- **功能:** 查看已入库的文档列表及状态
- **查询参数:**
  - `collection_name` (可选): 指定知识库名称
  - `status` (可选): 过滤状态 (processing/completed/failed)
  - `page` (可选): 页码，默认 1
  - `page_size` (可选): 每页数量，默认 20
- **响应格式:**
  ```json
  {
    "total": 100,
    "page": 1,
    "page_size": 20,
    "documents": [
      {
        "doc_id": "uuid-string",
        "filename": "example.pdf",
        "status": "completed",
        "chunks_count": 45,
        "created_at": "2024-01-01T00:00:00Z",
        "metadata": {}
      }
    ]
  }
  ```

#### 1.3 删除文档
- **接口:** `DELETE /v1/documents/{doc_id}`
- **功能:** 删除指定文档及其对应的向量索引
- **响应格式:**
  ```json
  {
    "doc_id": "uuid-string",
    "message": "文档及向量索引已删除",
    "deleted_chunks": 45
  }
  ```

### 2. 检索与问答

#### 2.1 向量检索
- **接口:** `POST /v1/rag/search`
- **功能:** 仅检索，返回最相关的 K 个知识片段（包含相似度得分）
- **请求格式:**
  ```json
  {
    "query": "用户问题",
    "collection_name": "default",  // 可选
    "top_k": 10,                   // 返回结果数量
    "score_threshold": 0.7,        // 可选，相似度阈值
    "use_hybrid": true             // 是否使用混合检索
  }
  ```
- **响应格式:**
  ```json
  {
    "query": "用户问题",
    "results": [
      {
        "chunk_id": "uuid-string",
        "content": "检索到的文本片段",
        "score": 0.95,
        "metadata": {
          "doc_id": "uuid-string",
          "filename": "example.pdf",
          "chunk_index": 12
        }
      }
    ],
    "total": 10
  }
  ```

#### 2.2 完整问答
- **接口:** `POST /v1/rag/chat`
- **功能:** 结合检索内容和用户问题，由 LLM 生成回答（支持流式输出）
- **请求格式:**
  ```json
  {
    "query": "用户问题",
    "collection_name": "default",
    "stream": false,               // 是否流式输出
    "temperature": 0.7,           // LLM 温度参数
    "max_tokens": 1000,           // 最大生成 token 数
    "top_k": 5,                   // 检索返回的文档数量
    "use_rerank": true            // 是否使用重排序
  }
  ```
- **响应格式 (非流式):**
  ```json
  {
    "answer": "LLM 生成的回答",
    "sources": [
      {
        "chunk_id": "uuid-string",
        "content": "引用的文本片段",
        "score": 0.95,
        "metadata": {}
      }
    ],
    "usage": {
      "prompt_tokens": 500,
      "completion_tokens": 200,
      "total_tokens": 700
    }
  }
  ```
- **响应格式 (流式):** Server-Sent Events (SSE) 格式

## 🧮 核心算法设计

### 1. 文档切片 (Chunking)

采用 **RecursiveCharacterTextSplitter** 策略，确保语义完整性：

- **Chunk Size:** 500 tokens (约 375 个中文字符)
- **Chunk Overlap:** 50 tokens (约 37 个中文字符，保持上下文连贯)
- **Separators:** `["\n\n", "\n", "。", "！", "？", " ", ""]` (按优先级递归分割)
- **Metadata 保留:** 每个 chunk 保留文档 ID、文件名、chunk 索引、位置信息等元数据

### 2. 向量化 (Embedding)

- **模型选择:** 支持多种 Embedding 模型，默认使用 `text-embedding-3-small` (1536 维)
- **批量处理:** 文档切片后批量生成向量，提升处理效率
- **向量归一化:** 对生成的向量进行 L2 归一化，提升检索精度

### 3. 混合检索 (Hybrid Retrieval)

混合检索结合了**向量相似度检索**和**关键词匹配检索**，提升检索效果：

#### 3.1 向量检索 (Dense Retrieval)
- 使用余弦相似度计算查询向量与文档向量的相似度
- Qdrant 使用 HNSW 算法进行快速近似最近邻搜索

#### 3.2 关键词检索 (Sparse Retrieval)
- 使用 BM25 算法进行关键词匹配
- 对查询和文档进行分词（支持中文分词）
- 计算 BM25 得分

#### 3.3 混合得分计算
```
最终得分 = α × 向量相似度得分 + (1 - α) × BM25 得分
```
- **α (alpha):** 混合权重，默认 0.7（向量检索权重）
- **得分归一化:** 两种得分分别归一化到 [0, 1] 区间后再加权合并

### 4. 重排序 (Rerank)

- **模型:** 使用 Cross-Encoder 模型对检索结果进行重排序
- **流程:** 检索出的前 10 条结果 → Cross-Encoder 重排序 → 取 Top-3 喂给 LLM
- **优势:** 提升检索精度，减少无关信息干扰 LLM

### 5. 提示词工程 (Prompt Engineering)

RAG 问答的提示词模板：
```
基于以下上下文信息回答用户问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文信息：
{context}

用户问题：{query}

请提供准确、简洁的回答：
```

## 📊 数据模型

### Qdrant Collection 结构

```python
{
    "collection_name": "default",
    "vectors": {
        "size": 1536,  # Embedding 维度
        "distance": "Cosine"  # 距离度量方式
    },
    "payload": {
        "doc_id": "uuid-string",      # 文档 ID
        "chunk_id": "uuid-string",    # Chunk ID
        "chunk_index": 12,            # Chunk 在文档中的索引
        "content": "文本内容",         # 原始文本
        "filename": "example.pdf",    # 文件名
        "metadata": {}                # 自定义元数据
    }
}
```

## ⚙️ 配置说明

### 环境变量

创建 `.env` 文件或设置环境变量：

```bash
# Qdrant 配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=default

# Embedding 模型配置
EMBEDDING_MODEL=openai  # openai / dashscope / ollama
EMBEDDING_MODEL_NAME=text-embedding-3-small
OPENAI_API_KEY=your-openai-api-key
DASHSCOPE_API_KEY=your-dashscope-api-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# LLM 配置
LLM_PROVIDER=openai  # openai / dashscope / ollama
LLM_MODEL_NAME=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Rerank 配置
USE_RERANK=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# 检索配置
DEFAULT_TOP_K=10
HYBRID_SEARCH_ALPHA=0.7  # 混合检索权重
RERANK_TOP_K=3  # Rerank 后返回的数量

# 文档处理配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_FILE_SIZE=50MB  # 最大文件大小

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/your-repo/RAG-FastAPI-Service.git
cd RAG-FastAPI-Service

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动 Qdrant

```bash
# 使用 Docker 启动 Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 验证 Qdrant 是否启动成功
curl http://localhost:6333/health
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写必要的配置：

```bash
cp .env.example .env
# 编辑 .env 文件，填入 API Key 等配置
```

### 4. 启动服务

```bash
# 开发模式启动
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式启动
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. 验证服务

访问 API 文档：`http://localhost:8000/docs`

## 📦 部署说明

### Docker 部署

```bash
# 构建镜像
docker build -t rag-fastapi-service .

# 运行容器
docker run -d \
  --name rag-service \
  -p 8000:8000 \
  --env-file .env \
  rag-fastapi-service
```

### Docker Compose 部署

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  rag-service:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - qdrant

volumes:
  qdrant_storage:
```

## 🔍 错误处理

### 常见错误码

- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源不存在
- `422 Unprocessable Entity`: 文件格式不支持或处理失败
- `500 Internal Server Error`: 服务器内部错误
- `503 Service Unavailable`: 依赖服务（Qdrant/LLM）不可用

### 错误响应格式

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "文档不存在",
    "details": {}
  }
}
```

## 📈 性能指标

### 预期性能

- **文档上传处理:** 平均 1000 tokens/秒
- **向量检索:** P99 延迟 < 100ms (10K 文档规模)
- **混合检索:** P99 延迟 < 200ms
- **RAG 问答:** P99 延迟 < 3s (包含 LLM 生成时间)

### 监控指标

- API 请求量、延迟、错误率
- Qdrant 查询性能
- LLM API 调用次数和成本
- 文档处理队列长度

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 运行集成测试
pytest tests/integration/

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

## 📝 开发计划

- [ ] 支持更多文档格式 (Markdown, HTML, Excel)
- [ ] 支持多模态文档 (图片 OCR)
- [ ] 实现文档版本管理
- [ ] 支持增量更新文档
- [ ] 添加用户认证和权限管理
- [ ] 实现检索结果缓存
- [ ] 支持多租户隔离