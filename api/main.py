"""
api/main.py 是整个 RAG 系统的 FastAPI 服务入口，
负责将底层 RAG Pipeline 封装为可访问的 HTTP API。
它通过生命周期管理在启动时初始化检索与生成引擎，并
提供统一的 REST 接口，包括问答查询（/query）、文档上传与索引（/upload、/index/batch）、
系统状态与健康检查（/health、/status）、以及 RAG 质量评估（/evaluate）等功能。
同时结合中间件、后台任务和流式响应机制，实现了高并发、可扩展的生产级 API 服务，是连接前端/用户请求与后端 RAG 能力的核心控制层。

FastAPI Application for RAG Service

这个文件是整个 RAG 系统的 API 入口（Controller 层）
负责把 RAG Pipeline 暴露成 HTTP 服务

核心功能：
- /query          → RAG 问答
- /upload         → 文档上传 & 索引
- /index/batch    → 批量索引
- /health         → 健康检查
- /evaluate       → RAG 质量评估
- /status         → 系统状态
- /documents/{id} → 删除文档
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# =========================
# 1. 路径配置（让 Python 能找到 src）
# =========================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =========================
# 2. FastAPI 相关依赖
# =========================
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# 日志工具
from loguru import logger

# 配置文件
import yaml

# =========================
# 3. 导入 RAG 核心模块
# =========================
from src.rag_pipeline import RAGPipeline
from src.data_loader import DataLoader, Document

# =========================
# 4. 导入 API 数据契约（schemas）
# =========================
from api.schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    IndexingStatus, HealthResponse, EvaluationRequest, EvaluationResponse,
    ErrorResponse
)

# =========================
# 5. 全局变量（RAG Pipeline 单例）
# =========================
pipeline: Optional[RAGPipeline] = None


# =========================
# 6. 配置加载函数
# =========================
def load_config():
    """
    加载系统配置文件 config/settings.yaml

    如果不存在：
    → 使用默认配置（保证系统可以运行）
    """
    config_path = os.path.join('config', 'settings.yaml')

    if not os.path.exists(config_path):
        # 默认配置（开发 / demo 用）
        return {
            'app': {'name': 'RAG API', 'version': '1.0.0'},
            'embedding': {
                'provider': 'openai',
                'model': 'text-embedding-ada-002'
            },
            'vector_store': {
                'provider': 'chroma',
                'chroma': {'persist_directory': './data/chroma_db'}
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4'
            }
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =========================
# 7. 生命周期管理（启动 / 关闭）
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理

    作用：
    - 启动时初始化 RAG Pipeline
    - 关闭时释放资源
    """
    global pipeline

    # ===== 启动阶段 =====
    logger.info("Starting up RAG API...")

    config = load_config()

    try:
        # 尝试从配置文件初始化完整 pipeline
        pipeline = RAGPipeline.from_config('config/settings.yaml')
        logger.info("RAG Pipeline initialized successfully")

    except Exception as e:
        # 如果失败，用最小配置兜底
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = RAGPipeline(config)

    yield  # ================= API 正常运行 =================

    # ===== 关闭阶段 =====
    logger.info("Shutting down RAG API...")


# =========================
# 8. 创建 FastAPI 应用实例
# =========================
app = FastAPI(
    title="RAG API",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# =========================
# 9. 中间件（性能 + 跨域）
# =========================

# 压缩响应（减少带宽）
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 允许跨域请求（前端调用 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 10. 根路径（健康信息）
# =========================
@app.get("/", response_model=HealthResponse)
async def root():
    """
    系统基本状态
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={
            "pipeline": "ready" if pipeline else "not_initialized",
            "vector_store": "connected" if pipeline else "unknown"
        }
    )


# =========================
# 11. 健康检查接口
# =========================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    更详细的健康检查
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "vector_store": f"{stats['vector_count']} documents",
            "embedding_model": f"{stats['embedding_dim']}d"
        }
    )


# =========================
# 12. RAG 核心接口（Query）
# =========================
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG 问答接口（核心 API）

    流程：
    1. 接收用户问题
    2. 检索相关文档（Retriever）
    3. LLM 生成回答（Generator）
    4. 返回答案 + sources
    """

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # =========================
        # 12.1 流式输出模式
        # =========================
        if request.stream:

            async def stream_generator():
                for chunk in pipeline.query(
                    question=request.question,
                    top_k=request.top_k,
                    stream=True
                ):
                    yield f"data: {chunk}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )

        # =========================
        # 12.2 普通模式（非流式）
        # =========================
        response = pipeline.query(
            question=request.question,
            top_k=request.top_k
        )

        return QueryResponse(
            answer=response.answer,
            sources=[
                {
                    "index": s["index"],
                    "text": s.get("text", ""),
                    "metadata": s["metadata"],
                    "relevance_score": s.get("score", 0.0)
                }
                for s in response.sources
            ],
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 13. 文档上传接口
# =========================
@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form("{}")
):
    """
    上传文档并进行索引

    支持格式：
    PDF / TXT / MD / DOCX / HTML
    """

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # 允许的文件类型
    allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.html', '.htm'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        # =========================
        # 13.1 保存临时文件
        # =========================
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # =========================
        # 13.2 加载文档
        # =========================
        document = pipeline.data_loader.load_document(tmp_path)

        # =========================
        # 13.3 合并 metadata
        # =========================
        import json
        try:
            custom_metadata = json.loads(metadata)
            document.metadata.update(custom_metadata)
        except json.JSONDecodeError:
            pass

        # =========================
        # 13.4 异步索引（后台任务）
        # =========================
        background_tasks.add_task(_index_document, document, tmp_path)

        return DocumentUploadResponse(
            status="processing",
            document_id=document.doc_id,
            chunks_indexed=0,
            message="Document uploaded and being indexed"
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 14. 后台索引任务
# =========================
def _index_document(document: Document, tmp_path: str):
    """
    后台执行：
    - 文档切分
    - 向量化
    - 写入 vector DB
    """
    try:
        pipeline.index_documents([document])
        logger.info(f"Indexed document: {document.doc_id}")

    finally:
        # 删除临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================
# 15. 批量索引接口
# =========================
@app.post("/index/batch")
async def batch_index(directory: str, background_tasks: BackgroundTasks):
    """
    批量索引目录中的所有文件
    """

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Invalid directory")

    background_tasks.add_task(pipeline.index_documents, directory)

    return {"status": "processing", "message": f"Indexing {directory}"}


# =========================
# 16. 系统状态
# =========================
@app.get("/status", response_model=IndexingStatus)
async def get_status():
    """
    返回向量库状态
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()

    return IndexingStatus(
        total_documents=stats['vector_count'],
        total_chunks=stats['vector_count'],
        status="ready"
    )


# =========================
# 17. RAG 评估接口
# =========================
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    对 RAG 输出进行质量评估：
    - faithfulness（忠实度）
    - relevance（相关性）
    - precision（精确度）
    """

    if not pipeline or not pipeline.evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    try:
        result = pipeline.evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        return EvaluationResponse(
            faithfulness=result.get('faithfulness', {}).get('score', 0),
            answer_relevancy=result.get('answer_relevancy', {}).get('score', 0),
            context_precision=result.get('context_precision', {}).get('score', 0),
            context_recall=result.get('context_recall', {}).get('score'),
            average_score=result.get('average_score', 0)
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 18. 删除文档接口
# =========================
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    从向量数据库删除文档
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.delete_document(doc_id)
        return {"status": "deleted", "document_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 19. 启动入口
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)