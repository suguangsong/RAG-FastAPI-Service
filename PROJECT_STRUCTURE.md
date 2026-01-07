# 项目结构说明

```
RAG-FastAPI-Service/
├── app/                          # 应用主目录
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── models.py                  # 数据模型定义
│   ├── routers/                   # API 路由
│   │   ├── __init__.py
│   │   ├── documents.py          # 文档管理路由
│   │   └── rag.py                # RAG 检索和问答路由
│   ├── services/                 # 业务逻辑服务
│   │   ├── __init__.py
│   │   ├── bm25_service.py       # BM25 关键词检索服务
│   │   ├── document_service.py   # 文档处理服务
│   │   ├── embedding_service.py # Embedding 向量化服务
│   │   ├── llm_service.py        # LLM 生成服务
│   │   ├── rerank_service.py     # 重排序服务
│   │   ├── retrieval_service.py  # 混合检索服务
│   │   └── vector_store.py       # Qdrant 向量存储服务
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── document_parser.py    # 文档解析器
│       └── text_splitter.py      # 文本切片器
├── scripts/                       # 脚本目录
│   ├── install_qdrant.sh        # Qdrant 安装脚本
│   ├── start_qdrant.sh          # Qdrant 启动脚本
│   └── stop_qdrant.sh           # Qdrant 停止脚本
├── main.py                       # 应用入口
├── requirements.txt              # Python 依赖
├── env.example                   # 环境变量模板
├── Dockerfile                    # Docker 镜像构建文件
├── docker-compose.yml            # Docker Compose 配置
├── run.sh                        # 启动脚本
├── README.md                     # 项目文档
└── LICENSE                       # 许可证

```

## 核心模块说明

### 1. 配置管理 (app/config.py)
- 使用 Pydantic Settings 管理所有配置项
- 支持从环境变量读取配置

### 2. 数据模型 (app/models.py)
- 定义所有 API 请求和响应的数据模型
- 使用 Pydantic 进行数据验证

### 3. 文档处理流程
1. **文档解析** (app/utils/document_parser.py)
   - 支持 PDF、DOCX、TXT 格式
2. **文本切片** (app/utils/text_splitter.py)
   - RecursiveCharacterTextSplitter
   - 支持 Token 计数和重叠切片
3. **向量化** (app/services/embedding_service.py)
   - 支持 OpenAI、DashScope、Ollama
4. **存储** (app/services/vector_store.py)
   - Qdrant 向量数据库存储

### 4. 检索流程
1. **混合检索** (app/services/retrieval_service.py)
   - 向量检索 + BM25 关键词检索
   - 加权合并得分
2. **重排序** (app/services/rerank_service.py)
   - Cross-Encoder 模型重排序
3. **LLM 生成** (app/services/llm_service.py)
   - 支持流式和非流式输出
   - 支持多种 LLM 提供商

### 5. API 路由
- `/v1/ingest/upload`: 上传文档
- `/v1/documents`: 查询文档列表
- `/v1/documents/{doc_id}`: 删除文档
- `/v1/rag/search`: 向量检索
- `/v1/rag/chat`: RAG 问答

