"""
EmbeddingFinetuner — Embedding 模型微调模块

参考论文：Arzideh et al., JMIR 2026 (MIRACLE)
损失函数：MultipleNegativesRankingLoss
  - batch 内其他样本自动作为负样本，无需显式构造
  - 适合 (query, document) 对的检索任务
"""

from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger


class EmbeddingFinetuner:
    """
    基于 sentence-transformers 的 Embedding 微调器

    用法：
        finetuner = EmbeddingFinetuner(
            base_model="BAAI/bge-small-zh-v1.5",
            output_path="models/medical-bge-finetuned"
        )
        finetuner.train(train_pairs, epochs=1, batch_size=32)
        finetuner.evaluate(test_pairs)
    """

    def __init__(
        self,
        base_model: str = "BAAI/bge-small-zh-v1.5",
        output_path: str = "models/medical-bge-finetuned",
        device: str = "cpu"
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self.base_model_name = base_model
        self.output_path = output_path
        self.device = device
        logger.info(f"EmbeddingFinetuner initialized: {base_model} → {output_path}")

    def train(
        self,
        train_pairs: List[Tuple[str, str]],
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
    ) -> None:
        """
        微调 Embedding 模型。

        Args:
            train_pairs: (question, chunk) 对列表
            epochs: 训练轮数（论文建议 1 epoch，防止过拟合）
            batch_size: 批大小（CPU 建议 32-64，GPU 可用 1024）
            learning_rate: 学习率（论文使用 2e-5）
            warmup_steps: 学习率预热步数
        """
        from sentence_transformers import SentenceTransformer, InputExample
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from torch.utils.data import DataLoader

        logger.info(f"Loading base model: {self.base_model_name}")
        model = SentenceTransformer(self.base_model_name, device=self.device)

        # 构造训练样本：(问题, chunk) → InputExample
        train_examples = [
            InputExample(texts=[question, chunk])
            for question, chunk in train_pairs
        ]
        logger.info(f"Training samples: {len(train_examples)}")

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )

        # MultipleNegativesRankingLoss：batch 内自动构造负样本
        # 原理：对 N 个 (q, d+) 对，每个 q 的负样本是 batch 中其他 N-1 个 d
        train_loss = MultipleNegativesRankingLoss(model)

        total_steps = len(train_dataloader) * epochs
        logger.info(
            f"Training config: epochs={epochs}, batch_size={batch_size}, "
            f"lr={learning_rate}, total_steps={total_steps}"
        )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=self.output_path,
            show_progress_bar=True,
        )

        logger.info(f"Model saved to: {self.output_path}")

    def evaluate(
        self,
        test_pairs: List[Tuple[str, str]],
        top_k: int = 10
    ) -> dict:
        """
        评估微调后模型的检索性能。

        指标：
        - Recall@K：top-K 中正确文档的召回率
        - MRR@K：平均倒数排名

        Args:
            test_pairs: (question, chunk) 对列表
            top_k: 评估截止值

        Returns:
            包含 recall@k 和 mrr@k 的字典
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np

        logger.info(f"Evaluating model: {self.output_path}")
        model = SentenceTransformer(self.output_path, device=self.device)

        questions = [q for q, _ in test_pairs]
        chunks = [c for _, c in test_pairs]

        # 向量化所有问题和 chunk
        logger.info("Encoding questions and chunks...")
        q_embeddings = model.encode(questions, show_progress_bar=True, batch_size=64)
        c_embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)

        # 计算余弦相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(q_embeddings, c_embeddings)

        recall_at_k = 0.0
        mrr_at_k = 0.0

        for i in range(len(questions)):
            # 第 i 个问题的正确文档是第 i 个 chunk
            scores = sim_matrix[i]
            ranked_indices = np.argsort(scores)[::-1][:top_k]

            if i in ranked_indices:
                recall_at_k += 1.0
                rank = list(ranked_indices).index(i) + 1
                mrr_at_k += 1.0 / rank

        n = len(questions)
        results = {
            f"recall@{top_k}": recall_at_k / n,
            f"mrr@{top_k}": mrr_at_k / n,
        }

        logger.info(f"Evaluation results: {results}")
        return results

    def compare_with_baseline(
        self,
        test_pairs: List[Tuple[str, str]],
        top_k: int = 10
    ) -> dict:
        """
        对比微调前后的性能差异。
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        def _eval(model_name_or_path: str) -> dict:
            model = SentenceTransformer(model_name_or_path, device=self.device)
            questions = [q for q, _ in test_pairs]
            chunks = [c for _, c in test_pairs]
            q_emb = model.encode(questions, batch_size=64, show_progress_bar=False)
            c_emb = model.encode(chunks, batch_size=64, show_progress_bar=False)
            sim = cosine_similarity(q_emb, c_emb)
            recall = mrr = 0.0
            for i in range(len(questions)):
                ranked = np.argsort(sim[i])[::-1][:top_k]
                if i in ranked:
                    recall += 1.0
                    mrr += 1.0 / (list(ranked).index(i) + 1)
            n = len(questions)
            return {f"recall@{top_k}": recall / n, f"mrr@{top_k}": mrr / n}

        baseline = _eval(self.base_model_name)
        finetuned = _eval(self.output_path)

        logger.info(f"Baseline  ({self.base_model_name}): {baseline}")
        logger.info(f"Fine-tuned ({self.output_path}): {finetuned}")

        improvement = {
            k: finetuned[k] - baseline[k]
            for k in baseline
        }
        logger.info(f"Improvement: {improvement}")

        return {
            "baseline": baseline,
            "finetuned": finetuned,
            "improvement": improvement
        }
