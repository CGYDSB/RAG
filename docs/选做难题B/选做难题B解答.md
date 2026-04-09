# 选做方向 B 解答：垂直领域语义偏移与模型微调

**面试题**：
- 痛点：开源的 Embedding 模型（如 BGE 等）在通用语料上表现很好，但遇到贵公司的内部专有名词、缩写（如"XYZ-990a 模块"）时，向量表征极其糟糕。
- 挑战：除了业务同义词表外，如果在资源允许的情况下，你会如何利用业务数据对 Embedding 模型或 ReRank 模型进行微调（Fine-tuning）？请简述训练数据的构造方式和损失函数选择。

---

## 一、问题根源分析

开源 Embedding 模型（如 BGE、multilingual-e5-large）在通用语料上训练，存在两类语义偏移问题：

**1. 专有名词表征差**

内部设备型号（"XYZ-990a 模块"）、药品编号（"盐酸二甲双胍片 0.5g"）、诊断编码（"ICD-10 E11.9"）在通用语料中出现极少，模型对这些词的向量表示接近随机。查询"XYZ-990a 的额定电压"时，向量检索可能召回语义相近但型号不同的文档。

**2. 领域语义漂移**

通用模型对"心搏停止"和"心脏骤停"可能不认为是同义词；对"Hb"（血红蛋白缩写）和"血红蛋白"的向量距离可能很远。这导致语义等价的查询和文档无法被正确匹配。

---

## 二、训练数据构造方式

### 2.1 整体流程

参考 MIRACLE 论文（Arzideh et al., JMIR 2026）的方法，使用 LLM 从内部文档合成问答对作为训练数据：

```
内部文档库（PDF/DOCX）
         ↓
文档切分（450字符，80字符重叠）
         ↓
LLM 合成问答对（每个 chunk 生成 5 个问答对）
         ↓
自动过滤（格式校验 + 幻觉检测）
         ↓
（可选）数据脱敏
         ↓
训练数据：(问题, chunk) 对
```

### 2.2 文档切分

使用递归字符切分，最大 450 字符，重叠 80 字符：

```python
# src/finetune_data_generator.py
self.splitter = RecursiveCharacterSplitter(
    chunk_size=450,
    chunk_overlap=80,
    min_chunk_size=50
)
```

选择 450 字符而非更大值的原因：内部技术文档通常以简洁、信息密集的段落或列表形式记录，较小的 chunk 能让每个问答对聚焦于单一知识点，减少噪声。

### 2.3 LLM 合成问答对

对每个 chunk，调用 LLM 生成 5 个问答对：

```python
QA_GENERATION_PROMPT = """你是一个专业的文档分析专家。请基于以下文档片段，生成5个相关问题及对应答案。

要求：
1. 每个问题必须以问号结尾
2. 答案必须能从文档片段中直接找到依据，不得编造
3. 问题应覆盖文档中的关键信息（专有名词、参数、操作规程、定义等）
4. 避免询问行政信息（如姓名、日期、编号等无实质内容的问题）

文档片段：{chunk}
"""
```

**为什么用 LLM 合成而不是人工标注**：
- 人工标注成本极高，内部文档可能有数万份
- LLM 能理解文档语义，生成的问题覆盖专有名词和领域术语
- 论文验证：76% 的合成问答对语义准确且临床正确，信噪比足够支持有效微调

### 2.4 数据过滤

自动过滤规则（参考论文，论文过滤掉约 24% 的问答对）：

```python
def _filter_qa_pairs(pairs, chunk):
    valid = []
    for question, answer in pairs:
        # 规则1：问题以问号结尾
        if not (question.endswith("？") or question.endswith("?")):
            continue
        # 规则2：答案不为空
        if len(answer.strip()) < 5:
            continue
        # 规则3：答案关键词能在 chunk 中找到（防幻觉）
        answer_keywords = [w for w in answer[:20] if '\u4e00' <= w <= '\u9fff']
        if answer_keywords:
            found = sum(1 for kw in answer_keywords if kw in chunk)
            if found / len(answer_keywords) < 0.3:
                continue  # 答案与 chunk 重叠率过低，疑似幻觉
        # 规则4：过滤行政类问题
        if any(kw in question for kw in ["姓名", "日期", "编号"]):
            continue
        valid.append((question, answer))
    return valid
```

### 2.5 数据集划分

按文档来源（而非随机）划分训练/测试集，避免同一文档的数据同时出现在训练集和测试集（数据泄露）：

```python
# 按来源文件分组，整文件划分
by_source = defaultdict(list)
for r in records:
    by_source[r["metadata"]["source"]].append((r["question"], r["chunk"]))

sources = list(by_source.keys())
train_sources = sources[:int(len(sources) * 0.8)]
test_sources = sources[int(len(sources) * 0.8):]
```

---

## 三、损失函数选择

### 3.1 MultipleNegativesRankingLoss（推荐，当前项目实现）

训练数据格式：`(问题, chunk)` 正样本对，无需显式构造负样本。

**原理**：对 batch 中 N 个 `(q, d+)` 对，每个 query 的负样本是 batch 中其他 N-1 个文档：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{\text{sim}(q_i, d_i^+)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(q_i, d_j)/\tau}}$$

```python
# src/embedding_finetuner.py
from sentence_transformers.losses import MultipleNegativesRankingLoss

train_examples = [
    InputExample(texts=[question, chunk])
    for question, chunk in train_pairs
]
train_loss = MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,           # 论文建议单 epoch，防止过拟合
    warmup_steps=100,
    optimizer_params={"lr": 2e-5},
    output_path="models/medical-bge-finetuned",
)
```

**选择理由**：
- 不需要显式构造负样本，数据构造成本低
- batch 越大，负样本越多，效果越好（论文 batch size=1024）
- 适合 (query, document) 对的检索任务

### 3.2 TripletLoss（有人工标注负样本时）

需要 `(query, positive_doc, negative_doc)` 三元组：

$$\mathcal{L} = \max(0,\ \text{sim}(q, d^-) - \text{sim}(q, d^+) + \text{margin})$$

**适用场景**：有专家标注的"相关/不相关"文档对，精度更高但数据构造成本大。

### 3.3 CoSENTLoss（有相关性分级时）

需要 `(query, doc, score)` 三元组，score 为相关性分级（0/1/2）：

$$\mathcal{L} = \log\left(1 + \sum_{(i,j): y_i > y_j} e^{\lambda(\text{sim}(q, d_j) - \text{sim}(q, d_i))}\right)$$

**适用场景**：有人工评分的检索数据集，能利用分级信息。

---

## 四、当前项目实现

### 4.1 CLI 命令

```bash
# 第一步：生成合成问答对
python main.py generate-finetune-data \
    --source data/raw/ \
    --output data/finetune/qa_pairs.jsonl \
    --max-chunks 20   # 调试时先用小值

# 第二步：微调 Embedding 模型
python main.py finetune \
    --data data/finetune/qa_pairs.jsonl \
    --base-model BAAI/bge-small-zh-v1.5 \
    --output models/medical-bge-finetuned \
    --epochs 1 \
    --batch-size 32

# 第三步：替换配置中的模型路径
# 编辑 config/settings.yaml：model: "models/medical-bge-finetuned"

# 第四步：清库重新索引
python main.py index --source data/raw/ --clear
```

### 4.2 训练配置对比

| 参数 | 论文（MIRACLE） | 当前项目 |
|------|----------------|---------|
| 基础模型 | multilingual-e5-large | BAAI/bge-small-zh-v1.5 |
| batch size | 1024 | 32-64（CPU 限制） |
| 学习率 | 2e-5 | 2e-5 |
| 优化器 | AdamW | AdamW |
| epochs | 1 | 1 |
| 损失函数 | CachedMultipleNegativesRankingLoss | MultipleNegativesRankingLoss |
| 硬件 | H100 80GB × 1，8天 | CPU，数小时（小数据集） |

### 4.3 评估：微调前后对比

微调完成后自动对比基础模型和微调模型的检索性能：

```python
# src/embedding_finetuner.py — compare_with_baseline()
baseline = _eval("BAAI/bge-small-zh-v1.5")
finetuned = _eval("models/medical-bge-finetuned")
# 输出：Recall@10、MRR@10 的提升幅度
```

论文数据：mAP@100 从 0.14 提升至 0.27（提升约 93%）。

### 4.4 集成到 RAG 系统

微调后的模型直接替换 `settings.yaml` 中的模型路径，`SentenceTransformerEmbedding` 类无需修改：

```yaml
embedding:
  provider: "sentence-transformers"
  model: "models/medical-bge-finetuned"  # 本地微调模型路径
  device: "cpu"
```

---

## 五、其他方法与技术

### 5.1 业务同义词表（轻量级方案）

在微调资源不足时，可以维护一个领域同义词表，在查询时做词语替换：

```python
SYNONYMS = {
    "心脏骤停": ["心搏停止", "cardiac arrest"],
    "Hb": ["血红蛋白", "hemoglobin"],
    "XYZ-990a": ["XYZ990a", "XYZ 990a 模块"],
}

def expand_query(query: str) -> str:
    for term, synonyms in SYNONYMS.items():
        if term in query:
            query += " " + " ".join(synonyms)
    return query
```

**优点**：零训练成本，立即生效
**缺点**：需要人工维护，无法覆盖所有语义变体

### 5.2 Reranker 微调

除了 Embedding 模型，也可以微调 Cross-Encoder Reranker，对 (query, chunk) 对直接打相关性分数：

```python
from sentence_transformers import CrossEncoder

# 训练数据：(query, chunk, label) 三元组，label=1 相关，label=0 不相关
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model.fit(
    train_dataloader=train_dataloader,
    loss_fct=torch.nn.BCEWithLogitsLoss(),
    epochs=3,
)
```

**优点**：精度更高（逐对计算，不受向量空间限制）
**缺点**：推理慢（每个候选都要单独计算），通常只对 top-K 候选做精排

### 5.3 领域自适应预训练（Domain-Adaptive Pretraining）

在微调之前，先用内部文档做无监督的继续预训练（MLM 任务），让模型先"见过"领域词汇：

```python
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

# 用内部文档做 MLM 继续预训练
model = AutoModelForMaskedLM.from_pretrained("BAAI/bge-small-zh-v1.5")
# 训练后再做有监督微调
```

**适用场景**：内部文档词汇与通用语料差异极大（如高度专业化的工业术语）

### 5.4 数据增强

在合成问答对的基础上，可以做数据增强提升训练数据多样性：

- **回译**：中文 → 英文 → 中文，生成语义等价但表达不同的问题
- **同义词替换**：用领域同义词替换问题中的关键词
- **问题改写**：用 LLM 对同一 chunk 生成不同角度的问题

### 5.5 Hard Negative Mining

`MultipleNegativesRankingLoss` 使用 batch 内随机负样本，效果有限。Hard Negative Mining 专门找"难负样本"（语义相近但不相关的文档）：

```python
# 用基础模型检索，取 top-K 但排除正样本的结果作为难负样本
hard_negatives = retriever.retrieve(question, top_k=20)
hard_negatives = [c for c in hard_negatives if c.id != positive_chunk_id][:5]

# 构造三元组训练数据
InputExample(texts=[question, positive_chunk, hard_negative])
```

**效果**：比随机负样本显著提升模型区分能力，尤其对专有名词密集的场景。

---

## 六、风险与缓解措施

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| 合成数据幻觉 | LLM 生成约 24% 含幻觉的问答对 | 自动过滤 + 人工抽检 |
| 模板过拟合 | 单一机构文档格式固定，模型可能学到格式而非语义 | 限制 1 epoch，多样化文档来源 |
| 性能回退 | 微调后在通用场景性能下降 | 保留原始模型作为 fallback，A/B 测试后切换 |
| 数据隐私 | 内部文档含敏感信息 | 数据脱敏后再用于训练，模型部署在内网 |
