"""
这个 schemas.py 是整个 RAG API 的数据契约层（Contract Layer），
通过 Pydantic 定义了请求、响应、评估和错误的统一结构，让 FastAPI 自动完成数据校验、文档生成和类型约束，是前后端通信的“标准协议”。

Pydantic Models for API（API数据模型层）

本文件用于定义 FastAPI / REST API 的请求与响应结构：

主要作用：
1. 数据校验（validation）
2. 自动生成 Swagger 文档
3. 统一 API 输入输出格式
4. 提升类型安全性

👉 相当于 RAG 系统的“数据协议层”
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# =========================================================
# 查询请求模型（用户提问）
# =========================================================
class QueryRequest(BaseModel):
    """
    用户 query 请求体

    用于 /query 接口
    """

    # 用户问题（必填）
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question"
    )

    # 检索 top_k 个文档
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )

    # 是否流式返回（Streaming API）
    stream: bool = Field(
        default=False,
        description="Stream the response"
    )

    # 可选过滤条件（如 metadata filter）
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters"
    )

    class Config:
        # Swagger 示例（API 文档展示用）
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "top_k": 5,
                "stream": False
            }
        }


# =========================================================
# 单条检索来源结构
# =========================================================
class Source(BaseModel):
    """
    表示 RAG 返回的单个引用来源（chunk）
    """

    index: int  # 来源编号
    text: Optional[str] = None  # chunk文本（可选，流式模式可能不返回）
    metadata: Dict[str, Any]  # 文档元信息（来源、文件名等）
    relevance_score: float  # 相关性分数（retriever输出）


# =========================================================
# 查询响应模型
# =========================================================
class QueryResponse(BaseModel):
    """
    /query 接口返回结构
    """

    # LLM 最终答案
    answer: str

    # 引用来源列表（RAG可解释性核心）
    sources: List[Source]

    # 额外信息（token数、检索数、延迟等）
    metadata: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "RAG (Retrieval-Augmented Generation) is...",
                "sources": [
                    {
                        "index": 1,
                        "text": "RAG combines retrieval with generation...",
                        "metadata": {"source": "doc1.pdf"},
                        "relevance_score": 0.95
                    }
                ],
                "metadata": {
                    "retrieved_count": 5,
                    "total_tokens": 150
                }
            }
        }


# =========================================================
# 文档上传请求
# =========================================================
class DocumentUploadRequest(BaseModel):
    """
    上传文档请求体
    """

    # 可附加元数据（标签、来源、类型等）
    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional metadata"
    )


# =========================================================
# 文档上传响应
# =========================================================
class DocumentUploadResponse(BaseModel):
    """
    文档入库结果
    """

    status: str  # 成功 / 失败
    document_id: Optional[str] = None  # 文档ID
    chunks_indexed: int  # 切分后的 chunk 数量
    message: str  # 说明信息


# =========================================================
# 索引状态
# =========================================================
class IndexingStatus(BaseModel):
    """
    向量库索引状态
    """

    total_documents: int
    total_chunks: int
    status: str


# =========================================================
# 健康检查接口
# =========================================================
class HealthResponse(BaseModel):
    """
    /health 接口返回
    """

    status: str  # healthy / unhealthy
    version: str  # 系统版本
    components: Dict[str, str]  # 各模块状态


# =========================================================
# RAG评估请求
# =========================================================
class EvaluationRequest(BaseModel):
    """
    手动触发 RAG 评估
    """

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


# =========================================================
# RAG评估结果
# =========================================================
class EvaluationResponse(BaseModel):
    """
    多维度评估结果
    """

    faithfulness: float          # 忠实性
    answer_relevancy: float      # 相关性
    context_precision: float     # 精度
    context_recall: Optional[float] = None  # 召回
    average_score: float         # 综合评分


# =========================================================
# 错误返回结构
# =========================================================
class ErrorResponse(BaseModel):
    """
    API错误统一格式
    """

    error: str
    detail: Optional[str] = None
    code: int