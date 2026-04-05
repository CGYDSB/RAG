# 垂直领域复杂文档 RAG 系统

面向医疗领域内部技术文档的智能问答系统，基于检索增强生成（RAG）架构，支持混合检索、多轮对话、防幻觉控制与引用溯源。

---

## 项目特性

- 多格式文档解析：PDF / TXT / MD / DOCX / HTML，PDF 含中文乱码清洗与页码追踪
- 混合检索：向量检索（BGE）+ BM25 关键词检索，加权融合
- 防幻觉三层机制：相似度阈值过滤 + 无关问题拦截 + Prompt 约束
- 引用溯源：每个结论标注 `[[序号]]`，可追溯到原始文档页码
- 多轮对话：对话历史增量压缩，8K token 窗口分层预算管理
- 流式输出：支持 SSE 流式返回
- 评估体系：Faithfulness / Relevancy / Precision / Recall
- 生产部署：FastAPI + Docker Compose

---

## 项目结构

```
.
├── api/
│   ├── main.py              # FastAPI 接口（/query /upload /health）
│   └── schemas.py           # 请求/响应数据结构
├── src/
│   ├── rag_pipeline.py      # 核心编排层（主流程）
│   ├── data_loader.py       # 多格式文档加载
│   ├── document_splitter.py # 递归/语义/固定切分策略
│   ├── embedding_model.py   # Embedding 模型（OpenAI/BGE/HuggingFace）
│   ├── vector_store.py      # 向量数据库（ChromaDB/Qdrant/Redis）
│   ├── retriever.py         # 混合检索 + Reranker
│   ├── generator.py         # LLM 生成层（OpenAI/Azure）
│   └── evaluation.py        # RAG 评估指标
├── config/
│   ├── settings.yaml        # 系统配置
│   └── prompts.yaml         # Prompt 模板
├── data/
│   ├── raw/                 # 原始文档
│   └── chroma_db/           # 向量库持久化
├── main.py                  # CLI 入口
├── docker-compose.yml
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（DeepSeek 兼容 OpenAI 接口）
```

### 3. 导入文档并建立索引

```bash
python main.py index --source data/raw/
```

### 4. 命令行问答

```bash
python main.py chat
```

### 5. 启动 API 服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger 文档：http://localhost:8000/docs
```

### 6. Docker 部署

```bash
docker-compose up -d
```

---

## 核心配置

`config/settings.yaml` 关键参数：

```yaml
llm:
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"

embedding:
  model: "BAAI/bge-small-zh-v1.5"   # 本地运行，无需 API

retrieval:
  similarity_threshold: 0.5          # chunk 过滤阈值
  off_topic_threshold: 0.3           # 低于此值判定为无关问题，直接拒答
  hybrid_search:
    vector_weight: 0.7
    keyword_weight: 0.3

pipeline:
  keep_recent_turns: 2               # 完整保留最近 N 轮对话
  max_total_tokens: 7000             # 总 token 预算
```

---

## 系统架构

```
用户问题
   ↓
向量化（BGE）
   ↓
混合检索（向量 + BM25）→ Top-K chunks
   ↓
相似度过滤 → 无关问题检测（直接拒答）
   ↓
Token 预算控制 → 构造上下文
   ↓
对话历史压缩注入
   ↓
Prompt 构造（防幻觉 + 引用规则）
   ↓
LLM 生成（DeepSeek）
   ↓
返回答案 + 来源 + 页码
```

---

## 技术选型

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek-chat（兼容 OpenAI 接口） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | ChromaDB（本地）/ Qdrant（生产） |
| 检索 | 自研 HybridRetriever |
| API 框架 | FastAPI |
| 部署 | Docker Compose |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/query` | 问答（支持流式） |
| POST | `/upload` | 上传文档并索引 |
| GET | `/health` | 健康检查 |
| GET | `/stats` | 系统统计信息 |
