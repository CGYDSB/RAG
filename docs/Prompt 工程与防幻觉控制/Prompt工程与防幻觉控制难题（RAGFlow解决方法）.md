# Prompt 工程与防幻觉控制难题方案

> 基于 RAGFlow v0.24.0 源码分析，聚焦 Prompt 构建、防幻觉控制与引用溯源三个核心问题

---

## 一、问题全景

RAG 系统的幻觉来源有三类：

1. **召回为空**：没有相关文档，模型凭空编造
2. **召回不足**：文档有相关内容但不完整，模型"补全"了不存在的细节
3. **召回有噪声**：召回了不相关的片段，模型被误导

对应地，防幻觉需要在三个层面介入：**Prompt 约束**、**系统级空结果处理**、**引用溯源验证**。

---

## 二、Prompt 构建：如何把召回片段喂给模型

### 2.1 知识块格式化（`kb_prompt`）

**代码位置**：`rag/prompts/generator.py` → `kb_prompt()` L103

每个召回的 chunk 被格式化成结构化的树形文本，而不是直接拼接原文：

```python
def kb_prompt(kbinfos, max_tokens, hash_id=False):
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        cnt = "\nID: {}".format(i)           # 序号，用于后续引用
        cnt += draw_node("Title", get_value(ck, "docnm_kwd", "document_name"))  # 文档名
        cnt += draw_node("URL", ck['url'])   # 来源 URL（如有）
        for k, v in docs.get(doc_id, {}).items():
            cnt += draw_node(k, v)           # 文档元数据（作者/日期等）
        cnt += "\n└── Content:\n"
        cnt += get_value(ck, "content", "content_with_weight")  # 正文内容
        knowledges.append(cnt)
```

格式化后的单个 chunk 示例：

```
ID: 0
├── Title: 液压系统技术手册_v3.2.pdf
├── Author: 工程部
└── Content:
主泵额定压力为 350bar，最大流量 120L/min。
在超压保护阀开启压力设定为 380bar 时...
```

**设计逻辑**：
- `ID: i` 是引用锚点，模型生成 `[ID:0]` 时系统能精确定位到哪个 chunk
- 标题和元数据让模型知道内容来源，有助于判断可信度
- 树形结构比纯文本拼接更清晰，减少模型混淆不同来源的概率

### 2.2 Token 预算控制

```python
# rag/prompts/generator.py L120-135
used_token_count = 0
chunks_num = 0
for i, c in enumerate(knowledges):
    used_token_count += num_tokens_from_string(c)
    chunks_num += 1
    if max_tokens * 0.97 < used_token_count:
        # 超出预算：截断，记录警告
        knowledges = knowledges[:i]
        logging.warning(f"Not all the retrieval into prompt: {len(knowledges)}/{kwlg_len}")
        break
```

知识块最多占用 97% 的 token 预算，剩余 3% 留给系统 Prompt 和生成空间。超出预算的 chunks 被截断，而不是压缩——保证每个进入 Prompt 的 chunk 都是完整的。

### 2.3 完整 Prompt 组装

**代码位置**：`api/db/services/dialog_service.py` L654-670

```python
# 1. 知识块注入到 {knowledge} 占位符
kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)

# 2. 系统 Prompt（用户自定义，含 {knowledge} 占位符）
msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]

# 3. 引用指令追加到系统 Prompt 末尾（仅当有召回内容时）
prompt4citation = ""
if knowledges and prompt_config.get("quote", True):
    prompt4citation = citation_prompt()   # 加载 citation_prompt.md

# 4. 历史对话
msg.extend([{"role": m["role"], "content": m["content"]} for m in messages])

# 5. 发送给 LLM（系统 Prompt + 引用指令合并）
answer = await chat_mdl.async_chat(prompt + prompt4citation, msg[1:], gen_conf)
```

**关键设计**：引用指令（`citation_prompt.md`）是**追加**到系统 Prompt 末尾的，而不是独立的消息。这样模型在处理用户问题时，引用规则始终在上下文中。

### 2.4 默认系统 Prompt（`ask_summary.md`）

**代码位置**：`rag/prompts/ask_summary.md`

```markdown
Role: You're a smart assistant. Your name is Miss R.
Task: Summarize the information from knowledge bases and answer user's question.
Requirements and restriction:
  - DO NOT make things up, especially for numbers.
  - If the information from knowledge is irrelevant with user's question,
    JUST SAY: Sorry, no relevant information provided.
  - Answer with markdown format text.
  - Answer in language of user's question.
  - DO NOT make things up, especially for numbers.  ← 重复强调，针对数字幻觉

### Information from knowledge bases
{{ knowledge }}
```

注意 `DO NOT make things up, especially for numbers` 出现了**两次**——这是刻意的，数字幻觉（编造统计数据、参数值）是 RAG 系统最常见也最危险的幻觉类型。

---

## 三、防幻觉控制：召回不足时的系统设计

### 3.1 第一道防线：空结果直接返回预设回复

**代码位置**：`api/db/services/dialog_service.py` L648-653

```python
knowledges = kb_prompt(kbinfos, max_tokens)

if not knowledges and prompt_config.get("empty_response"):
    # 召回为空 + 用户配置了空结果回复 → 直接返回，不调用 LLM
    empty_res = prompt_config["empty_response"]
    yield {"answer": empty_res, "reference": kbinfos, "final": True}
    return
```

这是最硬的防线：**完全不让 LLM 参与**。用户可以在知识库配置中设置 `empty_response`，比如"抱歉，我没有找到相关信息，请联系客服。"当召回为空时直接返回这个文本，LLM 没有任何发挥空间。

### 3.2 第二道防线：充分性检查（Sufficiency Check）

**代码位置**：`rag/prompts/generator.py` → `sufficiency_check()` L915，`rag/prompts/sufficiency_check.md`

```markdown
# sufficiency_check.md
You are a information retrieval evaluation expert. Please assess whether the
currently retrieved content is sufficient to answer the user's question.

User question: {{ question }}
Retrieved content: {{ retrieved_docs }}

Output format (JSON):
{
    "is_sufficient": true/false,
    "reasoning": "Your reasoning",
    "missing_information": ["Missing info 1", "Missing info 2"]
}
```

```python
# rag/prompts/generator.py L915-926
async def sufficiency_check(chat_mdl, question: str, ret_content: str):
    return await gen_json(
        PROMPT_JINJA_ENV.from_string(SUFFICIENCY_CHECK).render(
            question=question,
            retrieved_docs=ret_content
        ),
        "Output:\n",
        chat_mdl
    )
```

充分性检查用 LLM 判断召回内容是否足够回答问题，返回结构化 JSON。如果 `is_sufficient=false`，`missing_information` 字段会列出缺失的信息点，供下一步查询扩展使用。

### 3.3 第三道防线：查询扩展重检索（Multi-Query Generation）

**代码位置**：`rag/prompts/generator.py` → `multi_queries_gen()` L928，`rag/prompts/multi_queries_gen.md`

当充分性检查判定召回不足时，自动生成补充查询：

```markdown
# multi_queries_gen.md
You are a query optimization expert.
The user's original query failed to retrieve sufficient information;
please generate multiple complementary improved questions and corresponding queries.

Missing information: {{ missing_info }}

Output format (JSON):
{
    "reasoning": "Explanation of query generation strategy",
    "questions": [
        {"question": "Improved question 1", "query": "Improved query 1"},
        ...
    ]
}

Requirements:
4. Each query MUST be in the same language as the retrieved content.
5. DO NOT generate question and query that is similar to the original query.
```

生成 2-3 个不同角度的补充查询，重新检索，把新召回的内容合并进来再生成答案。

### 3.4 第四道防线：Prompt 级别的拒答指令

`ask_summary.md` 中的明确指令：

```
If the information from knowledge is irrelevant with user's question,
JUST SAY: Sorry, no relevant information provided.
```

即使召回了内容，如果内容与问题无关，模型被明确要求说"没有相关信息"，而不是强行用不相关内容拼凑答案。

### 3.5 第五道防线：引用约束（Citation Constraint）

**代码位置**：`rag/prompts/citation_prompt.md`

```markdown
## Technical Rules:
- DO NOT cite content not from <context></context>
- Maximum 4 citations per sentence

## What MUST Be Cited:
1. Quantitative data: Numbers, percentages, statistics, measurements
2. Temporal claims: Dates, timeframes, sequences of events
3. Causal relationships: Claims about cause and effect
...
```

引用规则本身就是防幻觉机制：**要求模型对每个事实性陈述标注来源**。如果模型编造了一个数字，它无法为这个数字找到对应的 `[ID:x]`，这会暴露幻觉。同时，`DO NOT cite content not from <context>` 明确禁止模型引用不存在的来源。

### 3.6 查询改写（多轮对话消歧）

**代码位置**：`rag/prompts/generator.py` → `full_question()` L228，`rag/prompts/full_question_prompt.md`

```markdown
# full_question_prompt.md
## Task & Steps
1. Generate a full user question that would follow the conversation.
2. If the user's question involves relative dates, convert them into absolute dates.

## Requirements
- If the user's latest question is already complete, don't do anything.
- DON'T generate anything except a refined question.
```

多轮对话中，"它的参数是多少？"这类指代不明的问题会被改写为完整问题（如"液压泵 HPV-135 的额定压力参数是多少？"），避免因问题歧义导致召回偏差，进而引发幻觉。

```python
# api/db/services/dialog_service.py L549-552
if len(questions) > 1 and prompt_config.get("refine_multiturn"):
    questions = [await full_question(dialog.tenant_id, dialog.llm_id, messages)]
```

---

## 四、引用溯源：精准定位到文档和页面

### 4.1 两种引用插入模式

RAGFlow 支持两种引用插入方式，根据 LLM 是否自动生成引用标记来选择：

**模式一：LLM 自动生成引用（主动模式）**

LLM 在生成答案时直接输出 `[ID:0]`、`[ID:1]` 等标记。系统用正则解析：

```python
# api/db/services/dialog_service.py L683-690
CITATION_MARKER_PATTERN = re.compile(r'\[ID:(\d+)\]')
normalized_answer = normalize_arabic_digits(answer)

if CITATION_MARKER_PATTERN.search(normalized_answer):
    # LLM 已经生成了引用标记，直接解析
    for match in CITATION_MARKER_PATTERN.finditer(normalized_answer):
        i = int(match.group(1))
        if i < len(kbinfos["chunks"]):
            idx.add(i)
```

**模式二：后处理自动插入引用（被动模式）**

LLM 没有生成引用标记时，系统用混合相似度自动匹配：

```python
# api/db/services/dialog_service.py L681-688
if embd_mdl and not CITATION_MARKER_PATTERN.search(normalized_answer):
    answer, idx = retriever.insert_citations(
        answer,
        [ck["content_ltks"] for ck in kbinfos["chunks"]],
        [ck["vector"] for ck in kbinfos["chunks"]],
        embd_mdl,
        tkweight=1 - dialog.vector_similarity_weight,
        vtweight=dialog.vector_similarity_weight,
    )
```

### 4.2 `insert_citations` 的实现原理

**代码位置**：`rag/nlp/search.py` → `Dealer.insert_citations()` L178

**第一步：答案按句子切分**

```python
# 按句子边界切分（支持中英文和阿拉伯语标点）
pieces = re.split(
    r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])",
    answer
)
# 代码块内容不切分（保护 ``` 块）
```

**第二步：对每个句子计算与所有 chunks 的混合相似度**

```python
ans_v, _ = embd_mdl.encode(pieces_)   # 句子向量化

for i, a in enumerate(pieces_):
    sim, tksim, vtsim = self.qryr.hybrid_similarity(
        ans_v[i],          # 答案句子向量
        chunk_v,           # 所有 chunk 向量
        rag_tokenizer.tokenize(pieces_[i]).split(),  # 答案句子 tokens
        chunks_tks,        # 所有 chunk tokens
        tkweight=0.1,      # 引用匹配更依赖向量（语义）
        vtweight=0.9,
    )
```

**第三步：阈值过滤 + 自适应降阈**

```python
thr = 0.63  # 初始阈值
while thr > 0.3 and len(cites.keys()) == 0:
    for i, a in enumerate(pieces_):
        mx = np.max(sim) * 0.99
        if mx < thr:
            continue
        # 相似度超过阈值的 chunks 作为该句子的引用来源
        cites[idx[i]] = [str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx][:4]
    thr *= 0.8  # 找不到引用时降低阈值重试
```

如果初始阈值 0.63 找不到任何引用，阈值会逐步降低（×0.8）直到 0.3，确保尽可能找到引用来源。

**第四步：引用标记插入**

```python
res = ""
for i, p in enumerate(pieces):
    res += p
    if i in cites:
        for c in cites[i]:
            res += f" [ID:{c}]"   # 在句子末尾插入引用标记
```

### 4.3 引用修复（容错处理）

**代码位置**：`api/db/services/dialog_service.py` → `repair_bad_citation_formats()`

LLM 有时会生成格式错误的引用（如 `[ID:0, ID:5]` 而不是 `[ID:0][ID:5]`），系统会自动修复：

```python
answer, idx = repair_bad_citation_formats(answer, kbinfos, idx)
```

### 4.4 位置信息：定位到具体页面

每个 chunk 在索引时存储了精确的位置信息：

```python
# rag/prompts/generator.py → chunks_format()
{
    "chunk_id": "...",
    "content_with_weight": "...",
    "positions": chunk.get("positions", "position_int"),  # 页码 + 坐标
    "document_name": "...",
    "document_id": "...",
    "similarity": 0.85,
    "vector_similarity": 0.82,
    "term_similarity": 0.71,
}
```

`positions` 字段存储的是 `[页码, x1, x2, y1, y2]` 格式的坐标数组，前端可以用这个信息在 PDF 预览中高亮显示对应区域。

### 4.5 文档聚合（doc_aggs）

引用解析完成后，系统还会统计每篇文档被引用的次数，生成文档级别的聚合结果：

```python
# api/db/services/dialog_service.py L693-700
idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
if not recall_docs:
    recall_docs = kbinfos["doc_aggs"]
kbinfos["doc_aggs"] = recall_docs
```

最终返回给前端的 `reference` 对象包含：
- `chunks`：被引用的具体片段（含位置坐标）
- `doc_aggs`：被引用的文档列表（文档名 + 引用次数）

---

## 五、完整防幻觉流程图

```
用户提问
    │
    ▼
查询改写（full_question）
    │ 多轮对话消歧，指代词替换为完整表述
    ▼
混合检索（retrieval）
    │
    ├─ 召回为空 ──→ empty_response 直接返回（不调用 LLM）
    │
    ▼
充分性检查（sufficiency_check）
    │
    ├─ 不充分 ──→ 查询扩展（multi_queries_gen）──→ 补充检索 ──→ 合并结果
    │
    ▼
Prompt 组装
    │  kb_prompt：结构化格式（ID + 标题 + 元数据 + 内容）
    │  system prompt：明确禁止编造，尤其是数字
    │  citation_prompt：要求对事实性陈述标注来源
    │
    ▼
LLM 生成答案（流式）
    │
    ▼
decorate_answer（后处理）
    │
    ├─ LLM 已生成 [ID:x] ──→ 正则解析，验证 ID 合法性
    │
    └─ LLM 未生成引用 ──→ insert_citations
           │  按句子切分答案
           │  每句与所有 chunks 计算混合相似度
           │  阈值过滤（0.63 → 自适应降低到 0.3）
           │  插入 [ID:x] 标记
           ▼
    引用修复（repair_bad_citation_formats）
           │
           ▼
    返回 {answer, reference: {chunks（含位置坐标）, doc_aggs}}
```

---

## 六、代码位置速查表

| 功能 | 文件 | 函数/位置 |
|------|------|---------|
| 知识块格式化 | `rag/prompts/generator.py` | `kb_prompt()` L103 |
| Token 预算控制 | `rag/prompts/generator.py` | `kb_prompt()` L120 |
| 空结果直接返回 | `api/db/services/dialog_service.py` | L648–653 |
| 充分性检查 | `rag/prompts/generator.py` | `sufficiency_check()` L915 |
| 充分性检查 Prompt | `rag/prompts/sufficiency_check.md` | — |
| 查询扩展生成 | `rag/prompts/generator.py` | `multi_queries_gen()` L928 |
| 查询扩展 Prompt | `rag/prompts/multi_queries_gen.md` | — |
| 查询改写（多轮） | `rag/prompts/generator.py` | `full_question()` L228 |
| 查询改写 Prompt | `rag/prompts/full_question_prompt.md` | — |
| 引用指令 Prompt | `rag/prompts/citation_prompt.md` | — |
| 默认系统 Prompt | `rag/prompts/ask_summary.md` | — |
| Prompt 组装主流程 | `api/db/services/dialog_service.py` | L654–670 |
| 引用自动插入 | `rag/nlp/search.py` | `insert_citations()` L178 |
| 引用格式修复 | `api/db/services/dialog_service.py` | `repair_bad_citation_formats()` |
| 位置坐标格式化 | `rag/prompts/generator.py` | `chunks_format()` L40 |
| 文档聚合 | `api/db/services/dialog_service.py` | L693–700 |
