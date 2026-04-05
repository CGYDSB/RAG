"""
Retrieval Strategies

支持：
1. HybridRetriever — Dense + Sparse 混合检索
2. QueryExpander   — 查询扩展
3. Reranker        — Cross-Encoder 重排序
4. FusionRetriever — RAG-Fusion
5. AdvancedRetriever — 统一封装
"""

import numpy as np
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import re
from loguru import logger


# =========================================
# 1. 检索结果结构
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
# 2. 抽象基类
# =========================================
class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        pass


# =========================================
# 3. Hybrid Retriever
# =========================================
class HybridRetriever(BaseRetriever):

    def __init__(self, vector_store, embedding_model,
                 vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._keyword_index: Dict = {}
        self._doc_freq: Dict = defaultdict(int)
        self._total_docs: int = 0
        logger.info("HybridRetriever initialized")

    def build_keyword_index(self, documents: List[Dict]) -> None:
        for doc in documents:
            doc_id = doc["id"]
            tokens = self._tokenize(doc["text"])
            for token in tokens:
                self._keyword_index.setdefault(token, {})
                self._keyword_index[token][doc_id] = \
                    self._keyword_index[token].get(doc_id, 0) + 1
            self._total_docs += 1

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        stopwords = {"the", "a", "is", "are", "to", "and", "of"}
        return [t for t in text.split() if t not in stopwords and len(t) > 2]

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
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

        def max_score(results):
            scores = [r["score"] for r in results]
            return max(scores) if scores else 1.0

        v_max = max_score(vector_results) or 1.0
        k_max = max_score(keyword_results) or 1.0

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
# 4. Query Expander
# =========================================
class QueryExpander:

    def expand_multi_query(self, query: str, n: int = 3) -> List[str]:
        return [
            query,
            f"What is {query}?",
            f"Explain {query}",
            f"Details about {query}",
        ][:n]

    def generate_hyde(self, query: str) -> str:
        return f"Document explaining: {query}"


# =========================================
# 5. Reranker
# =========================================
class Reranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker loaded: {model_name}")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
            c.final_score = (c.final_score + c.rerank_score) / 2
        return sorted(chunks, key=lambda x: x.final_score, reverse=True)[:top_n]


# =========================================
# 6. RAG-Fusion
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
# 7. AdvancedRetriever（统一封装）
# =========================================
class AdvancedRetriever:

    def __init__(self, vector_store, embedding_model, config: Optional[Dict] = None):
        config = config or {}
        hybrid_cfg = config.get("hybrid_search", {})
        vector_weight = hybrid_cfg.get("vector_weight", 0.7)
        keyword_weight = hybrid_cfg.get("keyword_weight", 0.3)

        self.hybrid = HybridRetriever(
            vector_store, embedding_model,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
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
        if self.reranker and results:
            results = self.reranker.rerank(query, results, self.top_n)
        return results[:top_k]
