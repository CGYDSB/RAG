# retrieval 模块

负责向量存储和混合检索，是 RAG 系统的召回层。

---

## 文件结构

```
retrieval/
├── __init__.py       # 公共接口导出
├── vector_store.py   # 向量数据库
└── retriever.py      # 混合检索器
```

---

## vector_store.py — 向量数据库

| 类 | 说明 |
|----|------|
| `VectorRecord` | 向量记录结构，包含 `id`、`vector`、`text`、`metadata` |
| `BaseVectorStore` | 抽象基类，定义 `add` / `search` / `delete` / `get_by_id` / `get_by_ids` / `count` / `clear` 接口 |
| `ChromaVectorStore` | 本地持久化（开发推荐），无需额外服务 |
| `QdrantVectorStore` | 生产级向量库，支持高并发 |
| `RedisVectorStore` | 缓存型，适合实时场景 |
| `VectorStoreFactory` | 工厂类，根据 `settings.yaml` 的 `vector_store.provider` 创建 |

`get_by_ids()` 用于父子 chunk 召回时批量取出父块内容。

**配置示例**：

```yaml
vector_store:
  provider: "chroma"
  chroma:
    persist_directory: "./data/chroma_db"
    collection_name: "documents"
```

---

## retriever.py — 混合检索器

| 类 | 说明 |
|----|------|
| `RetrievedChunk` | 检索结果结构，包含 `vector_score`、`keyword_score`、`rerank_score`、`final_score` |
| `HybridRetriever` | 向量检索 + BM25 关键词检索双路融合 |
| `Reranker` | Cross-Encoder 精排（可选，默认关闭） |
| `FusionRetriever` | RAG-Fusion，多查询扩展 |
| `AdvancedRetriever` | 统一封装，含空结果降级重试 |

**HybridRetriever 特性**：

- jieba 中文分词 + 名词权重差异化（名词 TF 翻倍）
- BM25 评分（TF 饱和 + 文档长度归一化），可切换 TF-IDF
- 关键词索引 pickle 持久化（`data/keyword_index.pkl`），启动自动加载
- 融合公式：`final_score = 0.7 × vector_score + 0.3 × keyword_score`

**评分方式切换**（`settings.yaml`）：

```yaml
retrieval:
  hybrid_search:
    vector_weight: 0.7
    keyword_weight: 0.3
    keyword_method: "bm25"    # 或 "tfidf"
```

**Reranker 融合公式**：

$$\text{final\_score} = 0.3 \cdot s_{\text{hybrid}} + 0.7 \cdot \sigma(r)$$

sigmoid 归一化解决 Cross-Encoder logit 与混合检索分数量纲不一致问题。

---

## 公共接口

```python
from src.retrieval import (
    VectorStoreFactory,     # 创建向量库
    ChromaVectorStore,      # 直接使用 ChromaDB
    VectorRecord,           # 向量记录结构
    AdvancedRetriever,      # 混合检索器
    RetrievedChunk,         # 检索结果结构
)
```
