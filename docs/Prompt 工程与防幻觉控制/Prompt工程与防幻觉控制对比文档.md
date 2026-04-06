# Prompt 工程与防幻觉控制对比文档

参考来源：Prompt工程与防幻觉控制难题（RAGFlow解决方法）.md（基于 RAGFlow v0.24.0 源码分析）
对比对象：当前项目 `config/prompts.yaml` + `src/rag_pipeline.py`

---

## 一、防幻觉机制总览对比

| 防线 | 当前项目 | RAGFlow |
|------|----------|---------|
| 召回为空时 | pipeline 层返回固定文本 | `empty_response` 配置项，完全不调用 LLM |
| 召回不足时 | 无检测，直接送入 LLM | 充分性检查（`sufficiency_check`）+ 查询扩展重检索 |
| Prompt 约束 | 7条规则，禁止捏造 + 引用标注 | `ask_summary.md`，"DO NOT make things up" 重复两次，专门强调数字 |
| 引用格式 | `[[序号]]`，LLM 自行生成 | `[ID:x]`，LLM 生成或后处理自动插入（双模式） |
| 引用验证 | 无（不验证引用是否真实存在） | 正则解析 ID，验证 ID 合法性，格式修复 |
| 引用精度 | chunk 级（序号对应整个 chunk） | 句级（每个句子单独匹配最相关 chunk） |
| 位置溯源 | 页码（`metadata.page`） | 页码 + 坐标（`[页码, x1, x2, y1, y2]`），可高亮 |
| 多轮消歧 | 无（指代词不处理） | `full_question()` 改写，指代词替换为完整表述 |
| chunk 格式化 | 纯文本拼接 + `[[序号]]` 前缀 | 树形结构（ID + 标题 + 元数据 + 内容） |
| 数字幻觉专项 | 无专项强调 | "DO NOT make things up, especially for numbers" 重复两次 |

---

## 二、逐模块详细对比

### 2.1 Prompt 约束层

**当前项目**（`rag_assistant`）：

```
1. 只能基于下方"参考文档"中的内容回答，禁止凭空捏造任何信息
2. 每一个关键事实必须用 [[序号]] 标注来源
3. 如果参考文档信息不完整，可合理推断但需注明"以下为基于文档片段的推断"
4. 涉及剂量、操作规程、安全警告等医疗敏感内容，必须原文引用，不得改写
5. 用中文回答，专业术语保持原文
6. 只有明显闲聊才拒答，技术类问题尽力作答
7. [章节名] 前缀是结构信息，不需要在回答中重复展示
```

**RAGFlow**（`ask_summary.md` + `citation_prompt.md`）：

```
- DO NOT make things up, especially for numbers.（重复两次）
- If the information from knowledge is irrelevant, JUST SAY: Sorry, no relevant information provided.
- Answer with markdown format text.
- Answer in language of user's question.

引用规则（citation_prompt.md）：
- DO NOT cite content not from <context></context>
- Maximum 4 citations per sentence
- 必须引用的内容：数字/百分比/统计数据、时间/日期、因果关系...
```

**差距**：

1. 当前项目没有专门强调**数字幻觉**。数字是 RAG 系统最危险的幻觉类型（编造参数值、统计数据），RAGFlow 重复两次强调，当前项目没有提及
2. 当前项目的引用规则是"每一个关键事实"，比较模糊。RAGFlow 明确列出了**必须引用的内容类型**（数字、时间、因果关系等），更具操作性
3. 当前项目没有限制每句话的引用数量（RAGFlow 限制最多 4 个），可能导致 LLM 对一句话堆砌大量引用

---

### 2.2 chunk 格式化

**当前项目**（`_build_context_with_budget`）：

```
[[1]] [7. 复合语句]
函数定义定义一个用户自定义的函数对象...

[[2]] [5. 表达式]
lambda 表达式也可以创建函数...
```

纯文本拼接，`[[序号]]` 前缀 + 可选的章节名前缀。

**RAGFlow**（`kb_prompt`）：

```
ID: 0
├── Title: 液压系统技术手册_v3.2.pdf
├── Author: 工程部
└── Content:
主泵额定压力为 350bar，最大流量 120L/min。
```

树形结构，包含文档名、作者等元数据。

**差距**：

当前项目的 chunk 格式缺少文档名信息。LLM 看到 `[[1]]` 时不知道这段内容来自哪个文档，只能靠 section_path 推断。RAGFlow 的树形格式让 LLM 在生成答案时就能感知内容来源，有助于在多文档场景下区分不同来源的信息。

---

### 2.3 引用溯源精度

**当前项目**：

- 引用粒度：chunk 级，`[[1]]` 对应整个 chunk（最多 256 字符）
- 引用生成：LLM 自行决定在哪里插入 `[[序号]]`
- 引用验证：无，LLM 可以引用不存在的序号（如 `[[6]]` 但只有 5 个 chunk）
- 位置信息：页码（`metadata.page`）

**RAGFlow**：

- 引用粒度：句级，每个句子单独匹配最相关 chunk
- 引用生成：双模式（LLM 主动生成 `[ID:x]` 或后处理自动插入）
- 引用验证：正则解析 ID，验证 ID < chunk 总数，格式修复
- 位置信息：页码 + 坐标（`[页码, x1, x2, y1, y2]`），前端可高亮

**差距**：

当前项目没有引用验证，LLM 可能生成 `[[6]]` 但实际只有 5 个 chunk，这是一种隐性幻觉（引用了不存在的来源）。RAGFlow 的后处理引用插入（`insert_citations`）通过混合相似度自动匹配，精度更高，且不依赖 LLM 的引用生成能力。

---

### 2.4 召回不足时的处理

**当前项目**：

召回不足时直接送入 LLM，LLM 会基于不完整的 context 生成答案，并标注"以下为基于文档片段的推断"。没有机制检测召回是否充分，也没有自动补充检索。

**RAGFlow**：

```
充分性检查 → 不充分 → 查询扩展 → 补充检索 → 合并结果 → 再生成
```

充分性检查用 LLM 判断召回内容是否足够，返回 `{is_sufficient, missing_information}`。如果不充分，自动生成 2-3 个补充查询，重新检索，把新召回的内容合并进来。

**差距**：

这是当前项目与 RAGFlow 差距最大的地方。当前项目对"召回不足"完全没有感知，只能靠 LLM 自己判断并标注推断。RAGFlow 的充分性检查 + 查询扩展是一个完整的闭环，能主动发现并弥补召回缺口。

---

### 2.5 多轮对话消歧

**当前项目**：

多轮对话时直接把原始问题送入检索，不做任何改写。"它的参数是多少？"这类问题会直接检索"它的参数"，召回结果可能完全偏离。

**RAGFlow**（`full_question`）：

```
用户：液压泵的额定压力是多少？
助手：350bar。
用户：它的最大流量呢？
         ↓ full_question 改写
"液压泵的最大流量是多少？"
```

把指代词替换为完整表述后再检索，避免因问题歧义导致召回偏差。

---

## 三、当前架构可改进的地方

按优先级排序：

### 优先级 1：引用验证（防止引用幻觉）

当前 LLM 可能生成超出范围的引用序号（如只有 5 个 chunk 但生成了 `[[6]]`）。在 `_generate_response` 里加一个简单的后处理验证：

```python
# src/rag_pipeline.py — _generate_response() 末尾
import re
answer_text = result.text
max_idx = len(chunks)
# 把超出范围的引用序号替换掉
def fix_citations(text, max_idx):
    def replace(m):
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= max_idx else ''
    return re.sub(r'\[\[(\d+)\]\]', replace, text)
answer_text = fix_citations(answer_text, max_idx)
```

### 优先级 2：数字幻觉专项强调

在 `rag_assistant` 和 `conversational` 的规则里加一条针对数字的专项约束：

```
8. 禁止编造任何数字、百分比、参数值、日期——这类信息必须原文引用，不得推断
```

数字幻觉是医疗场景最危险的错误类型，值得单独强调。

### 优先级 3：chunk 格式化加入文档名

在 `_build_context_with_budget` 里把文档名加入 chunk 展示：

```python
filename = chunk.metadata.get('filename', '') if chunk.metadata else ''
section = chunk.metadata.get('section_path', '') if chunk.metadata else ''
header = ' | '.join(filter(None, [filename, section]))
if header:
    text = f"[{header}]\n{text}"
```

让 LLM 在生成答案时能感知内容来自哪个文档，多文档场景下区分来源更准确。

### 优先级 4：多轮对话查询改写

在 `rag_pipeline.query()` 里，当有对话历史时，用 LLM 对当前问题做一次改写，把指代词替换为完整表述：

```python
# 有历史时，改写问题消除指代歧义
if use_history and self.conversation.history:
    question = self._rewrite_question(question)

def _rewrite_question(self, question: str) -> str:
    recent = self.conversation.format_recent_history()
    if not recent:
        return question
    prompt = f"""历史对话：\n{recent}\n\n当前问题：{question}\n\n
如果当前问题包含指代词（它、这个、该、上述等），请改写为完整问题；否则原样返回。
改写后的问题："""
    try:
        result = self.generator.generate(prompt, GenerationConfig(temperature=0, max_tokens=100))
        return result.text.strip() or question
    except:
        return question
```

### 优先级 5：引用每句限制数量

在 prompt 规则里加一条：

```
每个句子最多标注 3 个引用来源，避免堆砌引用
```

---

## 四、总结

当前项目的 Prompt 工程已经覆盖了基础的防幻觉需求（禁止捏造、引用标注、推断说明），但与 RAGFlow 相比，在以下三个方面存在明显差距：

1. **引用验证缺失**：LLM 可能生成不存在的引用序号，是一种隐性幻觉
2. **召回不足无感知**：没有充分性检查，召回不足时只能靠 LLM 自行标注推断
3. **多轮消歧缺失**：指代词不处理，多轮对话中检索质量会随轮次增加而下降

优先级 1（引用验证）和优先级 2（数字幻觉强调）改动量小、收益明显，建议优先实施。
