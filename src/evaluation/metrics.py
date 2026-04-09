"""
这个 evaluation.py 是整个 RAG 系统的评估核心模块，
用于从多个维度自动衡量模型效果，
包括答案是否真实依赖检索内容（faithfulness）、
回答是否与问题相关（relevancy）、检索内容是否有效（precision）以及是否覆盖正确知识（recall）。
它既支持 LLM 评估，也提供启发式 fallback 方法，并通过 RAGEvaluator 将所有指标统一管理，
实现单样本与批量评估，从而帮助开发者持续优化 RAG 系统质量。
RAG Evaluation Module（RAG评估模块）

本模块用于评估 RAG（Retrieval-Augmented Generation）系统效果，主要包含：

1. Faithfulness（忠实性）：答案是否基于检索内容，是否存在幻觉
2. Answer Relevancy（回答相关性）：回答是否真正回答了问题
3. Context Precision（上下文精度）：检索到的内容中有多少是相关的
4. Context Recall（上下文召回）：检索内容是否覆盖了标准答案信息
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger


# =========================
# 评估结果数据结构
# =========================
@dataclass
class EvaluationResult:
    """
    单个指标的评估结果封装类

    Attributes:
        metric_name: 指标名称
        score: 分数（0~1）
        details: 额外信息（如解释、错误分析等）
    """
    metric_name: str
    score: float
    details: Dict

    def to_dict(self) -> Dict:
        """转换为字典，方便 JSON 序列化或 API 返回"""
        return {
            'metric': self.metric_name,
            'score': self.score,
            'details': self.details
        }


# =========================
# 所有评估指标的抽象基类
# =========================
class BaseMetric(ABC):
    """
    评估指标抽象基类（所有 metric 必须实现 compute 方法）
    """

    @abstractmethod
    def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """计算指标分数"""
        pass


# =========================
# Faithfulness（忠实性指标）
# =========================
class FaithfulnessMetric(BaseMetric):
    """
    用于判断：
    👉 答案是否完全基于 context（是否 hallucination）

    思路：
    1. 如果有 LLM → 让 LLM 判断每个 claim
    2. 否则 → 用 token overlap 做简单启发式
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:

        # ===== 如果没有 LLM，则使用启发式方法 =====
        if not self.llm_client:
            return self._heuristic_faithfulness(answer, contexts)

        # ===== 使用 LLM 判断 =====
        context_text = "\n\n".join(contexts)

        prompt = f"""Evaluate if the answer is faithful to the context.

Context:
{context_text}

Answer: {answer}

Check each claim in the answer.
Is it supported by the context?

Return JSON:
{{
    "faithfulness_score": 0.0-1.0,
    "unsupported_claims": [],
    "explanation": ""
}}"""

        try:
            response = self.llm_client(prompt)

            import json
            result = json.loads(response)

            return EvaluationResult(
                metric_name="faithfulness",
                score=result.get("faithfulness_score", 0.0),
                details={
                    "unsupported_claims": result.get("unsupported_claims", []),
                    "explanation": result.get("explanation", "")
                }
            )

        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return self._heuristic_faithfulness(answer, contexts)

    def _heuristic_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """
        启发式方法：
        👉 计算 answer token 在 context 中的覆盖率
        """

        answer_tokens = set(answer.lower().split())
        context_tokens = set()

        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        if not answer_tokens:
            score = 0.0
        else:
            overlap = len(answer_tokens & context_tokens)
            score = overlap / len(answer_tokens)

        return EvaluationResult(
            metric_name="faithfulness",
            score=score,
            details={"method": "heuristic"}
        )


# =========================
# Answer Relevancy（回答相关性）
# =========================
class AnswerRelevancyMetric(BaseMetric):
    """
    判断：
    👉 answer 是否真正回答 question
    """

    def __init__(self, embedding_model: Optional[Callable] = None):
        self.embedding_model = embedding_model

    def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:

        # ===== 没有 embedding 模型时使用简单方法 =====
        if not self.embedding_model:
            return self._string_similarity(question, answer)

        try:
            # 向量化 question 和 answer
            question_emb = self.embedding_model.embed_query(question)
            answer_emb = self.embedding_model.embed_query(answer)

            similarity = self._cosine_similarity(question_emb, answer_emb)

            return EvaluationResult(
                metric_name="answer_relevancy",
                score=float(similarity),
                details={"method": "embedding_similarity"}
            )

        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return self._string_similarity(question, answer)

    def _string_similarity(self, question: str, answer: str) -> EvaluationResult:
        """
        简单 token overlap 相似度
        """

        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())

        if not q_tokens or not a_tokens:
            score = 0.0
        else:
            overlap = len(q_tokens & a_tokens)
            score = overlap / len(q_tokens)

        return EvaluationResult(
            metric_name="answer_relevancy",
            score=score,
            details={"method": "token_overlap"}
        )

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        a_arr = np.array(a)
        b_arr = np.array(b)

        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# =========================
# Context Precision（上下文精度）
# =========================
class ContextPrecisionMetric(BaseMetric):
    """
    衡量：
    👉 检索到的 context 有多少是“有用的”
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:

        if not contexts:
            return EvaluationResult(
                metric_name="context_precision",
                score=0.0,
                details={"relevant_count": 0, "total": 0}
            )

        # 使用 LLM 或启发式判断 relevance
        if self.llm_client:
            relevant_count = self._llm_judge_relevance(question, contexts)
        else:
            relevant_count = self._heuristic_relevance(question, contexts)

        score = relevant_count / len(contexts)

        return EvaluationResult(
            metric_name="context_precision",
            score=score,
            details={
                "relevant_count": relevant_count,
                "total": len(contexts)
            }
        )

    def _llm_judge_relevance(self, question: str, contexts: List[str]) -> int:
        """让 LLM 判断 context 是否相关"""
        relevant = 0

        for ctx in contexts:
            prompt = f"""
Is this context relevant to the question?

Question: {question}
Context: {ctx[:500]}

Answer ONLY YES or NO
"""

            try:
                response = self.llm_client(prompt).strip().upper()
                if response == "YES":
                    relevant += 1
            except:
                pass

        return relevant

    def _heuristic_relevance(self, question: str, contexts: List[str]) -> int:
        """基于关键词重叠的简单判断"""
        q_tokens = set(question.lower().split())
        relevant = 0

        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            if len(q_tokens & ctx_tokens) > 0:
                relevant += 1

        return relevant


# =========================
# Context Recall（上下文召回）
# =========================
class ContextRecallMetric(BaseMetric):
    """
    衡量：
    👉 context 是否覆盖 ground truth 信息
    """

    def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:

        if not ground_truth or not contexts:
            return EvaluationResult(
                metric_name="context_recall",
                score=0.0,
                details={"note": "ground_truth required"}
            )

        gt_tokens = set(ground_truth.lower().split())
        context_tokens = set()

        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        overlap = len(gt_tokens & context_tokens)
        score = overlap / len(gt_tokens) if gt_tokens else 0.0

        return EvaluationResult(
            metric_name="context_recall",
            score=score,
            details={
                "ground_truth_tokens": len(gt_tokens),
                "covered_tokens": overlap
            }
        )


# =========================
# RAG 统一评估器
# =========================
class RAGEvaluator:
    """
    一站式 RAG 评估框架
    👉 同时计算多个指标
    """

    def __init__(self, llm_client=None, embedding_model=None):
        self.metrics = {
            "faithfulness": FaithfulnessMetric(llm_client),
            "answer_relevancy": AnswerRelevancyMetric(embedding_model),
            "context_precision": ContextPrecisionMetric(llm_client),
            "context_recall": ContextRecallMetric()
        }

        self.results = []

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        单条样本评估
        """

        results = {}

        for name, metric in self.metrics.items():
            try:
                result = metric.compute(question, answer, contexts, ground_truth)
                results[name] = result.to_dict()
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results[name] = {
                    "metric": name,
                    "score": 0.0,
                    "error": str(e)
                }

        # 平均分
        scores = [r["score"] for r in results.values() if "score" in r]
        results["average_score"] = np.mean(scores) if scores else 0.0

        self.results.append({
            "question": question,
            "results": results
        })

        return results

    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """
        批量评估
        """

        all_results = []

        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth")
            )
            all_results.append(result)

        # 聚合统计
        aggregated = {}

        for metric in self.metrics.keys():
            scores = [r[metric]["score"] for r in all_results if metric in r]

            aggregated[metric] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }

        return {
            "aggregated": aggregated,
            "individual_results": all_results
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成评估报告
        """

        if not self.results:
            return "No evaluation results available"

        from collections import defaultdict
        metric_sums = defaultdict(list)

        for r in self.results:
            for metric_name, metric_result in r["results"].items():
                if isinstance(metric_result, dict) and "score" in metric_result:
                    metric_sums[metric_name].append(metric_result["score"])

        report_lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Total Evaluations: {len(self.results)}",
            ""
        ]

        for metric, scores in metric_sums.items():
            if scores:
                report_lines += [
                    f"{metric}:",
                    f"  Mean: {np.mean(scores):.3f}",
                    f"  Std:  {np.std(scores):.3f}",
                    f"  Min:  {np.min(scores):.3f}",
                    f"  Max:  {np.max(scores):.3f}",
                    ""
                ]

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report


# =========================
# 快速评估工具函数
# =========================
def evaluate_faithfulness(answer: str, contexts: List[str]) -> float:
    """快速计算 faithfulness"""
    metric = FaithfulnessMetric()
    return metric.compute("", answer, contexts).score


def evaluate_relevancy(question: str, answer: str) -> float:
    """快速计算 relevancy"""
    metric = AnswerRelevancyMetric()
    return metric.compute(question, answer, []).score