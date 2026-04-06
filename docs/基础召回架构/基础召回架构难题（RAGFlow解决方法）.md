# 基础召回架构难题方案

> 基于 RAGFlow v0.24.0 源码分析，聚焦混合检索（Hybrid Search）的设计与实现

---

## 一、核心问题

单路召回的天然缺陷：

- **纯向量检索**：擅长语义相似，但对精确术语、型号编号、专有名词不敏感。查询"液压泵型号 HPV-135"，向量检索可能召回语义相近但型号不同的文档。
- **纯关键词检索（BM25）**：精确匹配强，但无法理解同义词、近义词、语义变体。查询"发动机故障"无法召回描述"引擎异常"的文档。

混合检索的目标是让两路互补：**向量检索兜底语义，关键词检索保证精确**。

---

## 二、整体架构

RAGFlow 的混合检索分为三个阶段：

```
查询输入
    │
    ├─── 关键词检索（MatchTextExpr）─────────────────┐
    │    BM25 + 词权重 + 同义词扩展                   │
    │                                               ├──→ FusionExpr（weighted_sum）
    ├─── 向量检索（MatchDenseExpr）─────────────────┘        引擎层融合
    │    Embedding 余弦相似度 KNN                            weights: 0.05, 0.95
    │
    ↓
  引擎层初步融合结果（Top-N，默认 1024）
    │
    ↓
  应用层重排序（Rerank）
    │    ├─ 无 Rerank 模型：hybrid_similarity（向量 + token 加权求和）
    │    └─ 有 Rerank 模型：Cross-Encoder 精排 + token 相似度加权
    │
    ↓
  相似度阈值过滤 + 分页
    │
    ↓
  最终结果（附带 similarity / vector_similarity / term_similarity 三路分数）
```

---

## 三、第一阶段：引擎层融合（数据库内）

### 3.1 三个表达式对象

**代码位置**：`common/doc_store/doc_store_base.py`，`rag/nlp/search.py` L74

RAGFlow 用三个数据类描述检索意图，传给底层存储引擎：

```python
# common/doc_store/doc_store_base.py
class MatchTextExpr:    # 关键词检索
    fields: list[str]       # 检索字段（含权重，如 title_tks^10）
    matching_text: str      # 构造好的查询字符串
    topn: int
    extra_options: dict     # minimum_should_match 等

class MatchDenseExpr:   # 向量检索
    vector_column_name: str # 如 q_1024_vec
    embedding_data: list    # 查询向量
    distance_type: str      # cosine
    topn: int
    extra_options: dict     # similarity 阈值

class FusionExpr:       # 融合策略
    method: str             # "weighted_sum"
    topn: int
    fusion_params: dict     # {"weights": "0.05,0.95"}
```

### 3.2 构造融合查询

**代码位置**：`rag/nlp/search.py` → `Dealer.search()` L74

```python
# rag/nlp/search.py L120-135
matchText, keywords = self.qryr.question(qst, min_match=0.3)
matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))

# 权重：关键词 5%，向量 95%
fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
matchExprs = [matchText, matchDense, fusionExpr]

res = await thread_pool_exec(
    self.dataStore.search, src, highlightFields, filters,
    matchExprs, orderBy, offset, limit, idx_names, kb_ids
)
```

**权重设计逻辑**：引擎层给向量检索 95% 的权重，关键词检索只有 5%。这是因为引擎层的 BM25 分数和向量相似度量纲不同，直接加权会导致 BM25 分数压制向量分数。引擎层的关键词检索主要起**过滤**作用（`minimum_should_match=0.3`），真正的关键词权重在应用层重排序时补回来。

### 3.3 ES 的实现细节

**代码位置**：`rag/utils/es_conn.py` → `ESConnection.search()` L116

ES 用 `knn` + `query_string` 的组合实现混合检索：

```python
# rag/utils/es_conn.py L160-185
for m in match_expressions:
    if isinstance(m, MatchTextExpr):
        bool_query.must.append(Q("query_string",
            fields=m.fields,
            type="best_fields",
            query=m.matching_text,
            minimum_should_match=minimum_should_match,
            boost=1
        ))
        bool_query.boost = 1.0 - vector_similarity_weight  # 关键词部分降权

    elif isinstance(m, MatchDenseExpr):
        s = s.knn(
            m.vector_column_name,
            m.topn,
            m.topn * 2,
            query_vector=list(m.embedding_data),
            filter=bool_query.to_dict(),   # 关键词查询作为 KNN 的过滤条件
            similarity=similarity,
        )
```

关键设计：**关键词查询作为 KNN 的 filter**，而不是独立的召回路。这意味着向量检索的候选集已经被关键词过滤过，保证了精确性，同时用向量相似度做最终排序。

### 3.4 空结果降级策略

**代码位置**：`rag/nlp/search.py` L140-155

```python
# 第一次检索结果为空时，降低门槛重试
if total == 0:
    if filters.get("doc_id"):
        # 指定了文档 ID：直接返回该文档所有内容
        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], ...)
    else:
        # 降低 min_match 从 0.3 → 0.1，降低向量相似度阈值 0.1 → 0.17
        matchText, _ = self.qryr.question(qst, min_match=0.1)
        matchDense.extra_options["similarity"] = 0.17
        res = await thread_pool_exec(self.dataStore.search, ..., [matchText, matchDense, fusionExpr], ...)
```

---

## 四、第二阶段：关键词查询构造

这是 RAGFlow 混合检索的核心差异化能力，远不止简单的分词匹配。

**代码位置**：`rag/nlp/query.py` → `FulltextQueryer.question()` L30

### 4.1 检索字段权重

```python
# rag/nlp/query.py L33-41
self.query_fields = [
    "title_tks^10",        # 标题分词，权重 10
    "title_sm_tks^5",      # 标题细粒度分词，权重 5
    "important_kwd^30",    # 重要关键词，权重 30（索引时标注）
    "important_tks^20",    # 重要关键词分词，权重 20
    "question_tks^20",     # 问题分词，权重 20
    "content_ltks^2",      # 正文分词，权重 2
    "content_sm_ltks",     # 正文细粒度分词，权重 1
]
```

标题命中的权重是正文的 5-10 倍，`important_kwd` 的权重是正文的 15 倍。这保证了精确术语匹配时能排在前面。

### 4.2 词权重计算（IDF + NER + 词性）

**代码位置**：`rag/nlp/term_weight.py` → `Dealer.weights()`

每个词的权重由三个因素综合决定：

```python
# rag/nlp/term_weight.py
wts = (0.3 * idf1 + 0.7 * idf2) * np.array([ner(t) * postag(t) for t in tks])
```

| 因素 | 说明 |
|------|------|
| `idf1`（词频 IDF） | 基于词在语料中的出现频率，稀有词权重高 |
| `idf2`（文档频率 IDF） | 基于文档级别的 IDF，权重占 70% |
| `ner(t)`（命名实体） | 公司名/地名/股票代码权重 ×3，普通词 ×1 |
| `postag(t)`（词性） | 名词 ×2，地名/机构名 ×3，代词/连词 ×0.3 |

### 4.3 同义词扩展

**代码位置**：`rag/nlp/query.py` L60-80，`rag/nlp/synonym.py`

```python
# 对每个词查找同义词，加入查询但降权
syns = self.syn.lookup(tk)
tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
if tk_syns:
    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
```

同义词以 0.2 的权重加入查询，既扩大召回，又不影响精确匹配的排名。

### 4.4 Bigram 短语加权

```python
# rag/nlp/query.py L85-95
# 相邻词对构成短语，权重为两词中较大权重的 2 倍
for i in range(1, len(tks_w)):
    left, right = tks_w[i-1][0], tks_w[i][0]
    q.append('"%s %s"^%.4f' % (left, right, max(tks_w[i-1][1], tks_w[i][1]) * 2))
```

相邻词组成的短语权重翻倍，保证"液压泵"比单独的"液压"或"泵"排名更高。

---

## 五、第三阶段：应用层重排序

引擎层返回 Top-N（默认最多 1024 个候选），应用层再做精排。

**代码位置**：`rag/nlp/search.py` → `Dealer.retrieval()` L365，`Dealer.rerank()` L297

### 5.1 无 Rerank 模型：hybrid_similarity

**代码位置**：`rag/nlp/query.py` → `FulltextQueryer.hybrid_similarity()` L155

```python
def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([avec], bvecs)      # 向量余弦相似度
    tksim = self.token_similarity(atks, btkss)   # token 加权相似度
    return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]
```

默认权重：向量 70%，token 30%。这里的权重与引擎层的 5%/95% 不同——应用层有了归一化的分数，可以更均衡地融合。

**token_similarity 的实现**：

```python
# rag/nlp/query.py → token_similarity()
def to_dict(tks):
    d = defaultdict(int)
    wts = self.tw.weights(tks, preprocess=False)
    for i, (t, c) in enumerate(wts):
        d[t] += c * 0.4                          # 单词权重
        if i+1 < len(wts):
            d[t + wts[i+1][0]] += max(c, wts[i+1][1]) * 0.6  # bigram 权重
    return d
```

单词权重 40%，相邻词对（bigram）权重 60%，强调短语匹配。

### 5.2 rerank 时的字段加权

**代码位置**：`rag/nlp/search.py` → `Dealer.rerank()` L297

```python
for i in sres.ids:
    content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
    title_tks    = [t for t in sres.field[i].get("title_tks", "").split() if t]
    question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
    important_kwd = sres.field[i].get("important_kwd", [])

    # 标题词 ×2，重要关键词 ×5，问题词 ×6
    tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
    ins_tw.append(tks)
```

重排序时，标题词复制 2 份、重要关键词复制 5 份、问题词复制 6 份，相当于在 token 相似度计算中给这些字段更高的权重。

### 5.3 有 Rerank 模型：Cross-Encoder 精排

**代码位置**：`rag/nlp/search.py` → `Dealer.rerank_by_model()` L336

```python
def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3, vtweight=0.7, ...):
    # token 相似度（轻量）
    tksim = self.qryr.token_similarity(keywords, ins_tw)
    # Cross-Encoder 相似度（重量级，逐对计算）
    vtsim, _ = rerank_mdl.similarity(query, [" ".join(tks) for tks in ins_tw])
    # 加权融合 + PageRank 特征
    return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim
```

Cross-Encoder 对每个 (query, chunk) 对单独计算相关性，精度最高但计算量大，因此只对引擎层返回的 Top-N 做精排。

### 5.4 PageRank + Tag 特征加分

**代码位置**：`rag/nlp/search.py` → `Dealer._rank_feature_scores()` L270

```python
def _rank_feature_scores(self, query_rfea, search_res):
    # PageRank 分数（文档重要性）
    pageranks = np.array([search_res.field[id].get(PAGERANK_FLD, 0) for id in search_res.ids])

    # Tag 余弦相似度（内容标签匹配）
    for i in search_res.ids:
        for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
            if t in query_rfea:
                nor += query_rfea[t] * sc
            denor += sc * sc
        rank_fea.append(nor / sqrt(denor) / q_denor)

    return np.array(rank_fea) * 10. + pageranks
```

最终分数 = 混合相似度 + Tag 特征分 × 10 + PageRank 分。PageRank 默认权重为 10（`rank_feature={PAGERANK_FLD: 10}`），让高质量文档获得额外加分。

---

## 六、第四阶段：引用插入（Citation）

检索完成后，RAGFlow 还会把 LLM 生成的答案与召回的 chunks 做对齐，插入引用标记。

**代码位置**：`rag/nlp/search.py` → `Dealer.insert_citations()` L178

```python
# 把答案按句子切分
pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)

# 对每个句子计算与所有 chunks 的混合相似度
ans_v, _ = embd_mdl.encode(pieces_)
for i, a in enumerate(pieces_):
    sim, tksim, vtsim = self.qryr.hybrid_similarity(
        ans_v[i], chunk_v,
        rag_tokenizer.tokenize(pieces_[i]).split(),
        chunks_tks,
        tkweight=0.1, vtweight=0.9   # 引用匹配更依赖向量
    )
    if np.max(sim) >= thr:  # 阈值从 0.63 开始，不满足则降低到 0.8×thr
        cites[i] = [str(ii) for ii in range(len(chunk_v)) if sim[ii] > np.max(sim) * 0.99]
```

---

## 七、高级召回策略

### 7.1 TOC 辅助召回

**代码位置**：`rag/nlp/search.py` → `Dealer.retrieval_by_toc()` L608

对于有目录结构的文档，先用混合检索找到最相关的文档，再用 LLM 分析目录结构，找到最相关的章节，补充召回该章节的 chunks：

```python
# 找到相似度最高的文档
doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1)[0][0]
# 获取该文档的目录
toc = self.dataStore.search({"doc_id": doc_id, "toc_kwd": "toc"}, ...)
# 用 LLM 判断哪些目录条目与查询相关
ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
# 把 TOC 命中的 chunks 加入结果，相似度叠加
for cid, sim in ids:
    if cid in id2idx:
        chunks[id2idx[cid]]["similarity"] += sim  # 已有的 chunk 加分
    else:
        chunks.append(...)  # 新增 chunk
```

### 7.2 父子 Chunk 召回

**代码位置**：`rag/nlp/search.py` → `Dealer.retrieval_by_children()` L672

索引时可以把大 chunk 切成小 chunk（子 chunk），检索时命中子 chunk 后，自动替换为父 chunk 返回，保证上下文完整性：

```python
for id, cks in mom_chunks.items():
    chunk = self.dataStore.get(id, ...)  # 获取父 chunk 完整内容
    d = {
        "content_with_weight": chunk["content_with_weight"],  # 父 chunk 内容
        "similarity": np.mean([ck["similarity"] for ck in cks]),  # 子 chunk 相似度均值
        ...
    }
```

---

## 八、完整数据流

```
用户查询 "液压泵额定压力"
    │
    ├─ FulltextQueryer.question()
    │   ├─ 分词：["液压泵", "额定", "压力"]
    │   ├─ 词权重：液压泵(0.45) > 压力(0.35) > 额定(0.20)
    │   ├─ 同义词扩展：压力 → ["压强", "bar值"]（权重×0.2）
    │   ├─ Bigram：液压泵+额定(0.90), 额定+压力(0.70)
    │   └─ 构造 query_string → MatchTextExpr
    │
    ├─ emb_mdl.encode_queries("液压泵额定压力")
    │   └─ 1024 维向量 → MatchDenseExpr(cosine, topk=1024)
    │
    ├─ FusionExpr("weighted_sum", weights="0.05,0.95")
    │
    ↓ ES/Infinity 引擎执行（KNN + query_string filter）
    │   返回 Top-1024 候选
    │
    ├─ Dealer.rerank()（无 Rerank 模型）
    │   ├─ 向量余弦相似度（权重 0.7）
    │   ├─ token 加权相似度（权重 0.3）
    │   │   └─ 标题词×2, 重要词×5, 问题词×6
    │   └─ PageRank 加分
    │
    ├─ 相似度阈值过滤（默认 0.2）
    │
    └─ 返回 Top-K chunks（附 similarity/vector_similarity/term_similarity）
```

---

## 九、代码位置速查表

| 功能 | 文件 | 函数/位置 |
|------|------|---------|
| 混合检索主入口 | `rag/nlp/search.py` | `Dealer.retrieval()` L365 |
| 引擎层检索（含融合） | `rag/nlp/search.py` | `Dealer.search()` L74 |
| ES 混合检索实现 | `rag/utils/es_conn.py` | `ESConnection.search()` L116 |
| 关键词查询构造 | `rag/nlp/query.py` | `FulltextQueryer.question()` L30 |
| 混合相似度计算 | `rag/nlp/query.py` | `hybrid_similarity()` L155 |
| Token 相似度 | `rag/nlp/query.py` | `token_similarity()` L163 |
| 词权重计算（IDF+NER） | `rag/nlp/term_weight.py` | `Dealer.weights()` |
| 应用层重排序（无模型） | `rag/nlp/search.py` | `Dealer.rerank()` L297 |
| 应用层重排序（有模型） | `rag/nlp/search.py` | `Dealer.rerank_by_model()` L336 |
| PageRank + Tag 加分 | `rag/nlp/search.py` | `_rank_feature_scores()` L270 |
| 引用插入 | `rag/nlp/search.py` | `Dealer.insert_citations()` L178 |
| TOC 辅助召回 | `rag/nlp/search.py` | `retrieval_by_toc()` L608 |
| 父子 Chunk 召回 | `rag/nlp/search.py` | `retrieval_by_children()` L672 |
| 检索表达式定义 | `common/doc_store/doc_store_base.py` | `MatchTextExpr/MatchDenseExpr/FusionExpr` |
