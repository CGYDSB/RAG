# embedding 模块

负责文本向量化、Embedding 模型管理，以及垂直领域模型微调。

---

## 文件结构

```
embedding/
├── __init__.py     # 公共接口导出
├── model.py        # Embedding 模型（多后端支持）
├── finetuner.py    # Embedding 模型微调
└── data_gen.py     # 微调训练数据生成
```

---

## model.py — Embedding 模型

支持三种后端，通过工厂模式按配置创建：

| 类 | 说明 |
|----|------|
| `BaseEmbeddingModel` | 抽象基类，定义 `embed_documents` / `embed_query` / `dimension` 接口 |
| `OpenAIEmbedding` | OpenAI API（text-embedding-ada-002 等） |
| `SentenceTransformerEmbedding` | 本地模型（推荐，支持 BGE 系列），bge 模型检索时自动加前缀 |
| `HuggingFaceEmbedding` | HuggingFace 自定义模型 |
| `CachedEmbeddingModel` | LRU 缓存装饰器，避免重复向量化 |
| `EmbeddingModelFactory` | 工厂类，根据 `settings.yaml` 的 `embedding.provider` 创建对应实例 |

**配置示例**：

```yaml
embedding:
  provider: "sentence-transformers"
  model: "BAAI/bge-small-zh-v1.5"
  device: "cpu"
  cache_enabled: true
```

---

## finetuner.py — Embedding 模型微调

参考 MIRACLE 论文（Arzideh et al., JMIR 2026），使用 `MultipleNegativesRankingLoss` 对 Embedding 模型进行领域微调，解决专有名词向量表征差的问题。

| 方法 | 说明 |
|------|------|
| `train(train_pairs, epochs, batch_size)` | 微调训练，输入 `(问题, chunk)` 对列表 |
| `evaluate(test_pairs, top_k)` | 评估 Recall@K 和 MRR@K |
| `compare_with_baseline(test_pairs)` | 对比微调前后性能差异 |

**损失函数**：`MultipleNegativesRankingLoss`，batch 内其他样本自动作为负样本，无需显式构造。

**CLI 命令**：

```bash
python main.py finetune \
    --data data/finetune/qa_pairs.jsonl \
    --base-model BAAI/bge-small-zh-v1.5 \
    --output models/medical-bge-finetuned \
    --epochs 1 --batch-size 32
```

---

## data_gen.py — 微调训练数据生成

从内部文档库自动生成微调用的合成问答对。

| 类/函数 | 说明 |
|---------|------|
| `FinetuneDataGenerator` | 主生成器，加载文档 → 切分 → LLM 生成问答对 → 过滤 → 保存 JSONL |
| `load_qa_dataset(jsonl_path)` | 加载 JSONL，按文档来源划分训练/测试集（避免数据泄露） |

**生成流程**：

```
文档 → 切分（450字符/80重叠）→ LLM 生成 5 个问答对/chunk
     → 自动过滤（格式校验 + 幻觉检测）→ JSONL 文件
```

**过滤规则**：问题必须以问号结尾、答案不为空、答案关键词能在 chunk 中找到、过滤行政类问题。

**CLI 命令**：

```bash
python main.py generate-finetune-data \
    --source data/raw/ \
    --output data/finetune/qa_pairs.jsonl \
    --max-chunks 20   # 调试时先用小值
```

---

## 公共接口

```python
from src.embedding import (
    EmbeddingModelFactory,          # 创建 Embedding 模型
    SentenceTransformerEmbedding,   # 直接使用本地模型
    EmbeddingFinetuner,             # 微调
    FinetuneDataGenerator,          # 生成训练数据
    load_qa_dataset,                # 加载训练数据
)
```
