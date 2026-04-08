# Prompt 工程与防幻觉控制解答

**面试题**：Prompt 工程与防幻觉控制

1. 如何构建 Prompt 将召回的片段喂给大模型？
2. 如果召回的片段无法回答用户的问题，如何通过系统设计或 Prompt 限制模型"胡编乱造"？
3. 如何实现精准的"引用溯源（Citations）"，让用户知道答案来自哪份文档的哪一页？

---

## 问题一：如何构建 Prompt 将召回片段喂给大模型

### 1.1 chunk 格式化：前置文件名和章节名

召回的 chunk 在送入 Prompt 之前，先做结构化格式化，前置文件名和章节名，帮助 LLM 感知内容来源：

```python
# src/rag_pipeline.py — _build_context_with_budget()
filename = chunk.metadata.get('filename', '') if chunk.metadata else ''
section_path = chunk.metadata.get('section_path', '') if chunk.metadata else ''
header = ' | '.join(filter(None, [filename, section_path]))
if header:
    text = f"[{header}]\n{text}"
formatted = f"[[{i+1}]] {text}"
```

LLM 看到的每个 chunk 格式如下：

```
[[1]] [医疗设备操作手册.pdf | 3. 安全操作规程]
使用前必须检查设备密封性，确认无泄漏后方可开机...

[[2]] [药品说明书汇编.pdf | 阿司匹林]
本品用于解热镇痛，成人常用剂量为每次0.3-0.6g...
```

`[[序号]]` 是引用锚点，LLM 生成答案时用这个序号标注来源；`[文件名 | 章节名]` 让 LLM 知道内容归属，多文档场景下不会混淆不同来源的信息。

### 1.2 Token 预算控制：整体跳过低相关 chunk

召回的 chunk 按相关性分数降序排列，在 token 预算内尽量多放，超出预算时整体跳过（不截断单个 chunk，保证语义完整）：

```python
budget_tokens = self.token_budget['context']  # 4000 tokens
used_tokens = 0

for i, chunk in enumerate(chunks):
    chunk_tokens = self._count_tokens(formatted)
    if used_tokens + chunk_tokens > budget_tokens:
        break  # 整体跳过，不截断
    context_parts.append(formatted)
    used_tokens += chunk_tokens
```

**为什么不截断**：截断可能发生在句子中间，LLM 看到残缺信息反而更容易产生幻觉（用自己的知识"补全"）。整体跳过低相关 chunk，保证每个进入 Prompt 的 chunk 都是完整的。

### 1.3 完整 Prompt 结构

最终送给 LLM 的 Prompt 由三部分组成：

```
[系统角色 + 9条回答规则]
    ↓
【参考文档】
[[1]] [文件名 | 章节名]
chunk 内容...

[[2]] [文件名 | 章节名]
chunk 内容...
    ↓
【用户问题】
用户的具体问题
```

多轮对话时在参考文档前插入历史摘要：

```
[系统角色 + 规则]
    ↓
【对话历史摘要】
早期摘要 + 最近2轮完整对话
    ↓
【本轮参考文档】
[[1]] ...
    ↓
【用户问题】
...
```

---

## 问题二：如何防止模型"胡编乱造"

本项目采用四层防幻觉机制，从不同层面拦截幻觉：

### 第一层：pipeline 层拦截（最硬的防线）

在调用 LLM 之前，用 `vector_score` 判断问题是否与知识库相关：

```python
top_vector_score = retrieved_chunks[0].vector_score if retrieved_chunks else 0
if not filtered_chunks and top_vector_score < off_topic_threshold:
    return RAGResponse(
        answer="抱歉，您的问题超出了我的知识范围，我只能回答与文档相关的技术问题。",
        sources=[],
        metadata={'reason': 'off_topic'}
    )
```

**为什么用 `vector_score` 而不是 `final_score`**：混合检索的 `final_score` 在 BM25 关键词索引为空时会偏低，用语义相似度 `vector_score` 判断更可靠。

**效果**：完全无关的问题（如"今天天气怎么样"）直接短路返回，不调用 LLM，节省 API 费用，也避免 LLM 强行关联无关文档内容。

### 第二层：相似度阈值过滤

低相关的 chunk 不送入 LLM：

```python
threshold = config.get('retrieval', {}).get('similarity_threshold', 0.3)
filtered_chunks = [c for c in retrieved_chunks if c.final_score >= threshold]
```

即使通过了 pipeline 层拦截，低质量的 chunk 也会被过滤掉，减少 LLM 被噪声误导的概率。

### 第三层：Prompt 层约束（9条规则）

```
【回答规则】
1. 只能基于下方"参考文档"中的内容回答，禁止凭空捏造任何信息
2. 每一个关键事实必须用 [[序号]] 标注来源
3. 如果参考文档信息不完整，可合理推断但需注明"以下为基于文档片段的推断"
4. 涉及剂量、操作规程、安全警告等医疗敏感内容，必须原文引用，不得改写
5. 用中文回答，专业术语保持原文
6. 只有明显日常闲聊才拒答，技术类问题一律尽力作答
7. 参考文档中的 [章节名] 前缀是文档结构信息，不需要在回答中重复展示
8. 禁止编造任何数字、百分比、参数值、日期——这类信息必须原文引用，不得推断或估算
9. 每个句子最多标注 3 个引用来源，避免堆砌引用
```

**规则 8 专门针对数字幻觉**：数字是医疗场景最危险的幻觉类型——编造药品剂量、检验指标参考值、手术操作参数，可能直接危害患者安全。单独强调，要求数字类信息必须原文引用。

**规则 3 的设计逻辑**：不要求 LLM 在信息不足时直接拒答，而是允许合理推断但必须标注。这样既保证了回答的有用性，又让用户知道哪些内容是推断，可以自行核实。

### 第四层：后处理引用验证

LLM 可能生成超出范围的引用序号（如只有 5 个 chunk 但生成了 `[[6]]`），这是一种隐性幻觉——引用了不存在的来源：

```python
@staticmethod
def _fix_citations(text: str, max_idx: int) -> str:
    def replace(m):
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= max_idx else ''
    return re.sub(r'\[\[(\d+)\]\]', replace, text)

# _generate_response() 中调用
answer_text = self._fix_citations(result.text, len(chunks))
```

正则匹配所有 `[[数字]]`，序号超出 chunk 总数的直接删除。

### 四层机制总结

| 层级 | 触发时机 | 机制 | 效果 |
|------|----------|------|------|
| pipeline 层 | LLM 调用前 | vector_score 阈值判断 | 完全无关问题直接拒答，不调用 LLM |
| 检索层 | LLM 调用前 | final_score 阈值过滤 | 低相关 chunk 不进入上下文 |
| Prompt 层 | LLM 生成时 | 9条规则约束 | 禁止捏造、数字专项约束、推断标注 |
| 后处理层 | LLM 生成后 | 引用序号验证 | 过滤超出范围的引用，消除引用幻觉 |

---

## 问题三：如何实现精准的引用溯源

### 3.1 索引阶段：chunk 携带位置信息

文档加载时，每页插入 `[PAGE:N]` 标记，切分后每个 chunk 的 metadata 记录：

```python
metadata = {
    "filename": "医疗设备操作手册.pdf",   # 文件名
    "page": 10,                           # 页码（1-based）
    "section_path": "3. 安全操作规程",    # 章节名（来自 PDF 书签）
    "source": "data/raw/医疗设备操作手册.pdf",
    ...
}
```

`section_path` 来自 PDF 书签目录（Outline），通过页码区间映射：第5-11页属于"第3章 安全操作规程"，则这些页的 chunk 都携带该章节名。

### 3.2 生成阶段：LLM 用序号标注来源

Prompt 规则要求 LLM 对每个关键事实用 `[[序号]]` 标注来源，序号对应上下文中的 chunk 编号：

```
用户问：阿司匹林的禁忌症有哪些？

LLM 回答：
阿司匹林禁用于对水杨酸类药物过敏的患者 [[1]]。
活动性消化道溃疡患者禁用 [[2]]。
妊娠晚期妇女应避免使用，可能导致胎儿动脉导管早闭 [[1]][[2]]。
```

### 3.3 返回阶段：sources 携带完整位置信息

`_generate_response()` 构建 sources 列表，每条来源包含文件名、章节名、页码、相关度：

```python
sources = [
    {
        "index": i + 1,
        "text": c.text[:200],           # chunk 原文预览
        "metadata": c.metadata,         # 含 filename/page/section_path
        "score": c.final_score,
        "section_path": c.metadata.get('section_path', ''),
    }
    for i, c in enumerate(chunks)
]
```

### 3.4 展示阶段：精确到章节和页码

CLI 来源显示格式：

```
来源：
  [[1]] 医疗设备操作手册.pdf | 3. 安全操作规程 · 第10页 | 相关度 0.878
  [[2]] 药品说明书汇编.pdf   | 阿司匹林 · 第45页        | 相关度 0.780
```

用户看到 `[[1]]` 引用，可直接翻到对应文档的对应章节和页码核对原文。

### 3.5 溯源链路总结

```
索引阶段
  PDF 书签 → _extract_outline() → page_to_section 映射
  每页 [PAGE:N] 标记 → 切分后 chunk.metadata.page
  chunk.metadata = {filename, page, section_path, ...}
         ↓
生成阶段
  chunk 格式化：[[序号]] [文件名 | 章节名]\n内容
  LLM 在答案中插入 [[序号]] 标注
  _fix_citations() 验证序号合法性
         ↓
返回阶段
  sources[i] = {index, text, metadata, score, section_path}
         ↓
展示阶段
  [[1]] 文件名 | 章节名 · 第N页 | 相关度 X.XXX
```

每个 `[[序号]]` 都能精确追溯到：哪个文件 → 哪个章节 → 哪一页。
