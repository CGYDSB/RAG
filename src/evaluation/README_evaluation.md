# evaluation 模块

负责 RAG 系统质量评估，提供多维度指标计算。

---

## 文件结构

```
evaluation/
├── __init__.py     # 公共接口导出
└── metrics.py      # 评估指标实现
```

---

## metrics.py

| 类/函数 | 说明 |
|---------|------|
| `EvaluationResult` | 单个指标结果，包含 `metric_name`、`score`、`details` |
| `BaseMetric` | 抽象基类，定义 `compute(question, answer, contexts, ground_truth)` 接口 |
| `FaithfulnessMetric` | 忠实性：答案是否完全基于检索内容，无幻觉 |
| `AnswerRelevancyMetric` | 回答相关性：答案是否真正回答了问题 |
| `ContextPrecisionMetric` | 上下文精度：检索到的内容中有多少是有用的 |
| `ContextRecallMetric` | 上下文召回：检索内容是否覆盖了标准答案信息 |
| `RAGEvaluator` | 统一评估器，同时计算所有指标，支持单条和批量评估 |

**评估策略**：

- 有 LLM 客户端时：用 LLM 判断（精度高）
- 无 LLM 客户端时：用启发式方法（token overlap，速度快）

**使用方式**：

```python
evaluator = RAGEvaluator()
result = evaluator.evaluate(
    question="函数是什么？",
    answer="函数是用户自定义的函数对象...",
    contexts=["函数定义定义一个用户自定义的函数对象..."],
    ground_truth=None   # 可选，有则计算 context_recall
)
# result: {"faithfulness": {...}, "answer_relevancy": {...}, "average_score": 0.85}
```

**配置开关**（`settings.yaml`）：

```yaml
evaluation:
  enabled: false    # 默认关闭，开启后每次 query 都会评估
```

---

## 公共接口

```python
from src.evaluation import (
    RAGEvaluator,               # 统一评估器
    EvaluationResult,           # 评估结果结构
    FaithfulnessMetric,         # 单独使用忠实性指标
    AnswerRelevancyMetric,      # 单独使用相关性指标
)
```
