# 医疗领域内部技术文档智能问答系统（RAG）

面向医疗领域内部技术文档的高精度问答系统，基于检索增强生成（RAG）架构，参考 RAGFlow v0.24.0 源码对核心模块进行深度优化，支持混合检索、多轮对话、防幻觉控制与精准引用溯源。

---

## 核心特性

**文档解析与分块**
- 标题层级感知切片：提取 PDF 书签目录，每个 chunk 携带章节归属（`section_path`）
- 目录页过滤：自动识别并跳过目录页，避免目录条目干扰检索
- 碎片块合并：小于 50 字符的碎片块合并到前一块，避免无意义片段入库
- 父子分块：小块（256字符）精确检索，大块（1024字符）喂给 LLM 补充上下文
- 支持 PDF / TXT / MD / DOCX / HTML 五种格式，中文 PDF 专项清洗

**混合检索架构**
- 向量检索（BGE）+ jieba BM25 关键词检索双路融合（0.7/0.3 加权）
- BM25 评分：TF 饱和 + 文档长度归一化，可切换 TF-IDF
- 名词权重差异化：jieba 词性标注，名词 TF 翻倍，专有名词优先命中
- 关键词索引 pickle 持久化，启动自动加载，无需重新索引
- 空结果降级重试，Reranker sigmoid 归一化融合

**防幻觉控制**
- 9 条 Prompt 规则：禁止捏造、数字专项约束、引用数量限制
- 引用验证后处理：过滤超出 chunk 范围的引用序号
- 无关问题 pipeline 层拦截，不调用 LLM 直接返回
- 多轮对话含指代词时自动触发查询改写

**长上下文管理**
- 分层 Token 预算（7000 token 窗口，系统提示/历史/文档/问题/输出分层分配）
- 对话历史增量压缩（只压缩新增旧轮次，避免重复压缩）
- tiktoken 精确计数，动态生成预算，Prompt 超限三级降级兜底

**垂直领域 Embedding 微调**
- 参考 MIRACLE 论文（JMIR 2026），LLM 合成问答对 + MultipleNegativesRankingLoss
- 解决专有名词（型号、药品名）向量表征差的问题
- 完整 CLI 支持：数据生成 → 微调训练 → 评估对比 → 替换模型

**工程能力**
- FastAPI REST 接口，支持流式输出（SSE）
- Docker Compose 一键部署
- 评估体系：Faithfulness / Relevancy / Precision / Recall

---

## 项目结构

```
.
├── api/
│   ├── main.py              # FastAPI 接口（/query /upload /health）
│   └── schemas.py           # 请求/响应数据结构
├── src/
│   ├── pipeline.py          # 核心编排层（主流程）
│   ├── ingestion/           # 文档处理（加载 + 切分）
│   │   ├── loader.py        # 多格式文档加载 + PDF Outline + 目录页过滤
│   │   └── splitter.py      # 递归切分 + section_path + 碎片合并 + 父子分块
│   ├── retrieval/           # 检索层
│   │   ├── vector_store.py  # 向量数据库（ChromaDB/Qdrant/Redis）
│   │   └── retriever.py     # 混合检索（jieba BM25 + 向量）+ Reranker
│   ├── generation/          # 生成层
│   │   └── generator.py     # LLM 生成（OpenAI/Azure）+ Prompt 管理
│   ├── embedding/           # Embedding 层
│   │   ├── model.py         # Embedding 模型（OpenAI/BGE/HuggingFace）
│   │   ├── finetuner.py     # Embedding 模型微调
│   │   └── data_gen.py      # 微调训练数据生成
│   └── evaluation/          # 评估层
│       └── metrics.py       # RAG 评估指标
├── config/
│   ├── settings.yaml        # 系统配置
│   └── prompts.yaml         # Prompt 模板（9条防幻觉规则）
├── data/
│   ├── raw/                 # 原始文档
│   ├── chroma_db/           # 向量库持久化
│   ├── keyword_index.pkl    # 关键词索引持久化
│   └── finetune/            # 微调训练数据
├── models/                  # 微调后的模型（本地）
├── docs/                    # 技术文档
│   ├── 技术方案.md
│   ├── 修改日志.md
│   ├── 代码结构速览.md
│   ├── 基础召回架构/
│   ├── 文档解析与分块难题/
│   ├── Prompt 工程与防幻觉控制/
│   ├── 长上下文优化策略/
│   └── 选做难题B/           # Embedding 微调方案
├── main.py                  # CLI 入口
├── docker-compose.yml
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选：tiktoken 精确 token 计数
pip install tiktoken
# 可选：Embedding 微调
pip install sentence-transformers scikit-learn
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（DeepSeek 兼容 OpenAI 接口）
```

### 3. 导入文档并建立索引

```bash
# 首次索引
python main.py index --source data/raw/

# 清库重新索引
python main.py index --source data/raw/ --clear
```

索引完成后日志会显示：
```
INFO | Outline: 62 entries from xxx.pdf        # 书签提取（策略一）
INFO | TOC pages filtered: [2, 3]              # 目录页过滤（策略五）
INFO | Hierarchical split: 436 parent chunks   # 父子分块（如已开启）
INFO | Keyword index built: 1988 docs          # BM25 关键词索引
INFO | Keyword index saved: data/keyword_index.pkl
```

### 4. 命令行问答

```bash
python main.py chat                            # 基础对话
python main.py chat --show-sources             # 显示来源（文件 + 章节 + 页码 + 相关度）
python main.py chat --stream                   # 流式输出
python main.py chat --verbose --show-sources   # 显示检索原文（调试用）
```

### 5. Embedding 微调（可选）

```bash
# 生成合成问答对
python main.py generate-finetune-data --source data/raw/ --output data/finetune/qa_pairs.jsonl

# 微调模型
python main.py finetune --data data/finetune/qa_pairs.jsonl --output models/medical-bge-finetuned

# 替换配置中的模型路径后重新索引
# 编辑 settings.yaml: model: "models/medical-bge-finetuned"
python main.py index --source data/raw/ --clear
```

### 6. 启动 API 服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger 文档：http://localhost:8000/docs
```

### 7. Docker 部署

```bash
docker-compose up -d
```

---

## 系统架构

```
用户问题
   ↓
[多轮消歧] 含指代词时 LLM 改写为完整问题
   ↓
混合检索（向量 + jieba BM25）→ Top-K chunks
   ↓
相似度过滤 → 无关问题检测（直接拒答）
   ↓
父子 chunk：命中子块 → 取父块内容（上下文更完整）
   ↓
Token 预算控制 → 构造上下文（文件名 | 章节名 前缀）
   ↓
对话历史增量压缩注入
   ↓
Prompt 构造（9条防幻觉规则）
   ↓
LLM 生成（DeepSeek）→ 引用验证后处理
   ↓
返回答案 + 来源（文件 | 章节 · 页码 | 相关度）
```

---

## 关键配置

`config/settings.yaml` 核心参数：

```yaml
llm:
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"

embedding:
  model: "BAAI/bge-small-zh-v1.5"   # 或微调后的本地模型路径

document:
  default_chunk_size: 256
  default_chunk_overlap: 30
  splitting:
    hierarchical:
      enabled: false              # 开启父子分块
      parent_chunk_size: 1024

retrieval:
  similarity_threshold: 0.3
  off_topic_threshold: 0.3
  hybrid_search:
    vector_weight: 0.7
    keyword_weight: 0.3
    keyword_method: "bm25"        # 或 "tfidf"

pipeline:
  keep_recent_turns: 2
  max_total_tokens: 7000
```

---

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM | DeepSeek-chat | 兼容 OpenAI 接口，成本低，中文能力强 |
| Embedding | BAAI/bge-small-zh-v1.5 | 专为中文优化，本地运行，dim=512 |
| 向量数据库 | ChromaDB（本地）/ Qdrant（生产） | 本地开发无需额外服务 |
| 关键词检索 | jieba + BM25 | 中文分词 + TF 饱和 + 文档长度归一化 |
| Token 计数 | tiktoken（可选） | 精确计数，降级为字符估算 |
| API 框架 | FastAPI | 自动 Swagger，支持流式 |
| 部署 | Docker Compose | 一键部署 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/query` | 问答（支持流式） |
| POST | `/upload` | 上传文档并索引 |
| GET | `/health` | 健康检查 |
| GET | `/stats` | 向量库统计信息 |

---

## 模块文档

各子模块有独立的 README：

| 模块 | 文档 |
|------|------|
| 文档处理 | `src/ingestion/README_ingestion.md` |
| 检索层 | `src/retrieval/README_retrieval.md` |
| 生成层 | `src/generation/README_generation.md` |
| Embedding | `src/embedding/README_embedding.md` |
| 评估 | `src/evaluation/README_evaluation.md` |
| 代码结构总览 | `docs/代码结构速览.md` |
