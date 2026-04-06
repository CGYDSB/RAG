"""
Retrieval Strategies

优化（参考召回架构对比文档）：
1. HybridRetriever — 中文分词（jieba）+ 词权重差异化 + 关键词索引持久化
2. Reranker        — sigmoid 归一化后加权融合（替代简单平均）
3. AdvancedRetriever — 空结果降级重试
"""

import math
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import re
from loguru import logger


# =========================================
# 检索结果结构
# =========================================
@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.final_score,
        }


# =========================================
# 抽象基类
# =========================================
class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        pass


# =========================================
# Hybrid Retriever
# 优化一：中文分词（jieba）
# 优化二：词权重差异化（名词 ×2）
# 优化三：关键词索引持久化（pickle）
# =========================================
class HybridRetriever(BaseRetriever):

    # 中文停用词
    _CN_STOPWORDS = {"的", "了", "是", "在", "和", "与", "或", "等", "也", "都",
                     "就", "但", "而", "及", "对", "从", "到", "被", "把", "将"}
    # 英文停用词
    _EN_STOPWORDS = {"the", "a", "is", "are", "to", "and", "of", "in", "for"}

    def __init__(self, vector_store, embedding_model,
                 vector_weight: float = 0.7, keyword_weight: float = 0.3,
                 index_path: str = "data/keyword_index.pkl"):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.index_path = index_path
        self._keyword_index: Dict = {}
        self._total_docs: int = 0

        # 尝试加载持久化的关键词索引
        self._load_index()
        logger.info(f"HybridRetriever initialized (keyword_docs={self._total_docs})")

    # --------------------------------------------------
    # 优化一 + 二：中文分词 + 词权重差异化
    # --------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """
        中英文混合分词：
        - 中文：jieba 分词，名词权重 ×2（通过重复 token 实现）
        - 英文：空格切分
        - 过滤停用词和单字符
        """
        try:
            import jieba
            import jieba.posseg as pseg
            tokens = []
            for word, flag in pseg.cut(text):
                word = word.strip()
                if not word or word in self._CN_STOPWORDS or len(word) < 2:
                    continue
                tokens.append(word)
                # 名词（n开头）权重翻倍：重复加入一次
                if flag.startswith('n') and len(word) >= 2:
                    tokens.append(word)
            return tokens
        except ImportError:
            # jieba 未安装，降级为空格切分
            logger.warning("jieba not installed, falling back to whitespace tokenization")
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [t for t in text.split()
                    if t not in self._EN_STOPWORDS and len(t) > 2]

    def build_keyword_index(self, documents: List[Dict]) -> None:
        """构建关键词倒排索引"""
        self._keyword_index = {}
        self._total_docs = 0
        for doc in documents:
            doc_id = doc["id"]
            tokens = self._tokenize(doc["text"])
            for token in tokens:
                self._keyword_index.setdefault(token, {})
                self._keyword_index[token][doc_id] = \
                    self._keyword_index[token].get(doc_id, 0) + 1
            self._total_docs += 1
        logger.info(f"Keyword index built: {self._total_docs} docs, {len(self._keyword_index)} terms")

    # --------------------------------------------------
    # 优化三：索引持久化
    # --------------------------------------------------
    def save_index(self) -> None:
        """持久化关键词索引到磁盘"""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "index": self._keyword_index,
                "total_docs": self._total_docs
            }, f)
        logger.info(f"Keyword index saved: {self.index_path}")

    def _load_index(self) -> None:
        """从磁盘加载关键词索引"""
        if Path(self.index_path).exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                self._keyword_index = data.get("index", {})
                self._total_docs = data.get("total_docs", 0)
                logger.info(f"Keyword index loaded: {self._total_docs} docs, {len(self._keyword_index)} terms")
            except Exception as e:
                logger.warning(f"Failed to load keyword index: {e}")

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        if not self._keyword_index or self._total_docs == 0:
            return []
        tokens = self._tokenize(query)
        scores: Dict = defaultdict(float)
        for token in tokens:
            if token not in self._keyword_index:
                continue
            idf = np.log((self._total_docs + 1) / (len(self._keyword_index[token]) + 1))
            for doc_id, freq in self._keyword_index[token].items():
                scores[doc_id] += freq * idf
        return [
            {"id": k, "score": v}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_vec = self.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_vec, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        return self._fuse(vector_results, keyword_results, top_k)

    def _fuse(self, vector_results, keyword_results, top_k) -> List[RetrievedChunk]:
        chunks: Dict[str, RetrievedChunk] = {}

        v_max = max((r["score"] for r in vector_results), default=1.0) or 1.0
        k_max = max((r["score"] for r in keyword_results), default=1.0) or 1.0

        for r in vector_results:
            chunks[r["id"]] = RetrievedChunk(
                id=r["id"],
                text=r.get("text", ""),
                metadata=r.get("metadata", {}),
                vector_score=r["score"] / v_max
            )

        for r in keyword_results:
            if r["id"] in chunks:
                chunks[r["id"]].keyword_score = r["score"] / k_max
            else:
                chunks[r["id"]] = RetrievedChunk(
                    id=r["id"],
                    text=r.get("text", ""),
                    metadata={},
                    keyword_score=r["score"] / k_max
                )

        for c in chunks.values():
            c.final_score = (
                self.vector_weight * c.vector_score +
                self.keyword_weight * c.keyword_score
            )

        return sorted(chunks.values(), key=lambda x: x.final_score, reverse=True)[:top_k]


# =========================================
# Query Expander
# =========================================
class QueryExpander:

    def expand_multi_query(self, query: str, n: int = 3) -> List[str]:
        return [query, f"What is {query}?", f"Explain {query}", f"Details about {query}"][:n]

    def generate_hyde(self, query: str) -> str:
        return f"Document explaining: {query}"


# =========================================
# Reranker
# 优化四：sigmoid 归一化后加权融合（替代简单平均）
# =========================================
class Reranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker loaded: {model_name}")

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
            # 优化：sigmoid 归一化 Cross-Encoder 分数后加权融合，避免量纲不一致
            c.final_score = 0.3 * c.final_score + 0.7 * self._sigmoid(c.rerank_score)
        return sorted(chunks, key=lambda x: x.final_score, reverse=True)[:top_n]


# =========================================
# RAG-Fusion
# =========================================
class FusionRetriever:

    def retrieve(self, query: str, retriever: BaseRetriever, top_k: int = 5) -> List[RetrievedChunk]:
        queries = [query, f"Explain {query}", f"Details about {query}"]
        all_results = [retriever.retrieve(q, top_k * 2) for q in queries]
        return self._rrf(all_results)[:top_k]

    def _rrf(self, results_list: List[List[RetrievedChunk]]) -> List[RetrievedChunk]:
        scores: Dict = defaultdict(float)
        chunks: Dict = {}
        k = 60
        for results in results_list:
            for rank, chunk in enumerate(results):
                scores[chunk.id] += 1 / (k + rank)
                chunks[chunk.id] = chunk
        return sorted(chunks.values(), key=lambda x: scores[x.id], reverse=True)


# =========================================
# AdvancedRetriever（统一封装）
# 优化五：空结果降级重试
# =========================================
class AdvancedRetriever:

    def __init__(self, vector_store, embedding_model, config: Optional[Dict] = None):
        config = config or {}
        hybrid_cfg = config.get("hybrid_search", {})

        self.hybrid = HybridRetriever(
            vector_store, embedding_model,
            vector_weight=hybrid_cfg.get("vector_weight", 0.7),
            keyword_weight=hybrid_cfg.get("keyword_weight", 0.3),
        )
        self.expander = QueryExpander()

        reranker_cfg = config.get("reranker", {})
        self.reranker = None
        if reranker_cfg.get("enabled", False):
            try:
                self.reranker = Reranker(
                    model_name=reranker_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                )
            except Exception as e:
                logger.warning(f"Reranker init failed, disabled: {e}")

        self.top_n = reranker_cfg.get("top_n", 5)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        results = self.hybrid.retrieve(query, top_k * 2)

        # 优化五：空结果降级——扩大候选范围重试一次
        if not results:
            logger.debug(f"Empty results for query, retrying with top_k*4")
            results = self.hybrid.retrieve(query, top_k * 4)

        if self.reranker and results:
            results = self.reranker.rerank(query, results, self.top_n)

        return results[:top_k]
