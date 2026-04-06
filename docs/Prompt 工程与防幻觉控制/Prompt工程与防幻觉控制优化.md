# Prompt 工程与防幻觉控制优化

参考来源：Prompt工程与防幻觉控制对比文档.md — 三、当前架构可改进的地方
改动文件：`src/rag_pipeline.py`、`config/prompts.yaml`

---

## 一、优化总览

| 优先级 | 问题 | 优化方案 | 涉及文件 |
|--------|------|----------|----------|
| 1 | LLM 可能生成超出范围的引用序号（隐性幻觉） | `_fix_citations()` 后处理过滤非法引用 | `rag_pipeline.py` |
| 2 | 无数字幻觉专项约束 | prompt 规则 8：禁止编造数字/参数值/日期 | `prompts.yaml` |
| 3 | chunk 格式化缺少文档名 | `_build_context_with_budget` 前置文件名 + 章节名 | `rag_pipeline.py` |
| 4 | 多轮对话指代词不处理，检索质量随轮次下降 | `_rewrite_question()` 改写含指代词的问题 | `rag_pipeline.py` |
| 5 | 无引用数量限制，LLM 可能堆砌引用 | prompt 规则 9：每句最多 3 个引用 | `prompts.yaml` |

---

## 二、优化一：引用验证（防止引用幻觉）

### 问题

LLM 在生成答案时可能输出 `[[6]]`，但实际只有 5 个 chunk。这是一种隐性幻觉——引用了不存在的来源，用户无法追溯，且系统不会报错。

### 改动

在 `_generate_response()` 里，对 LLM 输出做后处理，过滤超出范围的引用序号：

```python
# src/rag_pipeline.py
@staticmethod
def _fix_citations(text: str, max_idx: int) -> str:
    import re
    def replace(m):
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= max_idx else ''
    return re.sub(r'\[\[(\d+)\]\]', replace, text)

# _generate_response() 中调用
answer_text = self._fix_citations(result.text, len(chunks))
```

正则匹配所有 `[[数字]]`，序号在 `[1, chunk总数]` 范围内的保留，超出范围的直接删除。

---

## 三、优化二：数字幻觉专项约束

### 问题

数字是 RAG 系统最危险的幻觉类型（编造参数值、统计数据、日期），原有规则只有通用的"禁止捏造"，没有专门针对数字的约束。

### 改动

`rag_assistant` 和 `conversational` 均新增规则：

```yaml
# rag_assistant 规则 8
8. 禁止编造任何数字、百分比、参数值、日期——这类信息必须原文引用，不得推断或估算

# conversational 规则 7
7. 禁止编造任何数字、百分比、参数值、日期，这类信息必须原文引用
```

参考 RAGFlow 的 `ask_summary.md` 中 `DO NOT make things up, especially for numbers` 重复两次的设计，对数字幻觉单独强调。

---

## 四、优化三：chunk 格式化加入文档名

### 问题

原有 chunk 格式只有章节名前缀，LLM 不知道内容来自哪个文档。多文档场景下，LLM 可能混淆不同文档的内容。

### 改动

`_build_context_with_budget()` 中，把文件名和章节名合并为 header 前置：

```python
# 改动前
section_path = chunk.metadata.get('section_path', '') if chunk.metadata else ''
if section_path:
    text = f"[{section_path}]\n{text}"

# 改动后
filename = chunk.metadata.get('filename', '') if chunk.metadata else ''
section_path = chunk.metadata.get('section_path', '') if chunk.metadata else ''
header = ' | '.join(filter(None, [filename, section_path]))
if header:
    text = f"[{header}]\n{text}"
```

LLM 看到的 chunk 格式变为：

```
改动前：
[[1]] [7. 复合语句]
函数定义定义一个用户自定义的函数对象...

改动后：
[[1]] [python-doc-27-34.pdf | 7. 复合语句]
函数定义定义一个用户自定义的函数对象...
```

---

## 五、优化四：多轮对话查询改写

### 问题

多轮对话中，"它的参数是多少？"这类含指代词的问题会直接送入检索，召回结果可能完全偏离。随着对话轮次增加，指代词问题越来越多，检索质量持续下降。

### 改动

在 `query()` 里，有对话历史且问题含指代词时，调用 `_rewrite_question()` 改写：

```python
# query() 中
retrieval_question = question
if use_history and self.conversation.history:
    retrieval_question = self._rewrite_question(question)
retrieved_chunks = self.retriever.retrieve(retrieval_question, top_k=top_k)
```

`_rewrite_question()` 的设计：

```python
def _rewrite_question(self, question: str) -> str:
    # 1. 简单判断：只有含指代词才调用 LLM，避免无谓的 API 消耗
    pronouns = ['它', '这个', '该', '上述', '前面', '之前', '其', '这些', '那个']
    if not any(p in question for p in pronouns):
        return question

    # 2. 用最近几轮对话作为上下文，让 LLM 改写
    recent = self.conversation.format_recent_history()
    prompt = f"历史对话：\n{recent}\n\n当前问题：{question}\n\n..."

    # 3. 改写失败时返回原始问题，不影响主流程
    try:
        result = self.generator.generate(prompt, GenerationConfig(temperature=0, max_tokens=100))
        return result.text.strip() or question
    except:
        return question
```

**关键设计**：
- 只有含指代词时才触发改写，避免每轮都多一次 LLM 调用
- `temperature=0` 保证改写结果稳定
- 改写结果过长时（超过原问题 3 倍）不采用，防止 LLM 过度扩展
- 改写失败时静默降级，不影响主流程

**效果示例**：

```
用户：液压泵的额定压力是多少？
助手：350bar [[1]]。
用户：它的最大流量呢？
         ↓ _rewrite_question 改写
检索用问题："液压泵的最大流量是多少？"
```

---

## 六、优化五：引用数量限制

### 问题

LLM 有时会对一个句子堆砌大量引用（如 `[[1]][[2]][[3]][[4]][[5]]`），降低可读性，且多数引用是冗余的。

### 改动

`rag_assistant` 和 `conversational` 均新增规则：

```yaml
# rag_assistant 规则 9
9. 每个句子最多标注 3 个引用来源，避免堆砌引用

# conversational 规则 8
8. 每个句子最多标注 3 个引用来源
```

---

## 七、改动文件汇总

| 文件 | 改动内容 |
|------|----------|
| `src/rag_pipeline.py` | 新增 `_fix_citations()`、`_rewrite_question()`；`_generate_response()` 调用引用验证；`query()` 调用查询改写；`_build_context_with_budget()` 加入文件名 |
| `config/prompts.yaml` | `rag_assistant` 新增规则 8、9；`conversational` 新增规则 7、8 |
