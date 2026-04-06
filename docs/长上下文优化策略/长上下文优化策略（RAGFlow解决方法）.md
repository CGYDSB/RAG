# 长上下文优化策略

> 基于 RAGFlow v0.24.0 源码分析，聚焦多轮对话 + 大量召回文档场景下的 Token 预算管理

---

## 一、问题建模

典型的超长上下文场景：

```
系统 Prompt（含知识块）：~6000 tokens
前 5 轮对话历史：~3000 tokens
本次用户问题：~200 tokens
引用指令：~800 tokens
─────────────────────────────
合计：~10000 tokens  >  8K 窗口
```

核心矛盾：**知识块（召回内容）和对话历史都很重要，但 Token 预算有限**。

RAGFlow 的解法不是单一机制，而是一套**分层预算管理 + 多级降级**策略。

---

## 二、Token 预算的来源与分配

### 2.1 max_tokens 的确定

**代码位置**：`api/db/services/dialog_service.py` L477

```python
max_tokens = llm_model_config.get("max_tokens", 8192)
```

`max_tokens` 直接读取模型配置中的上下文窗口大小。GPT-4o 是 128K，本地小模型可能只有 4K 或 8K。这个值是整个预算分配的基准。

### 2.2 预算分配优先级

```
max_tokens（模型上下文窗口）
    │
    ├─ 97% 用于知识块（kb_prompt 的硬上限）
    │
    └─ 95% 用于完整消息列表（message_fit_in 的硬上限）
           │
           ├─ 系统 Prompt（含知识块 + 引用指令）
           └─ 对话历史（剩余空间）
```

知识块和消息列表共享同一个 95% 的预算，知识块先占，对话历史用剩余空间。

---

## 三、第一层：知识块的 Token 预算控制

### 3.1 按相似度排序后截断

**代码位置**：`rag/prompts/generator.py` → `kb_prompt()` L103

```python
def kb_prompt(kbinfos, max_tokens, hash_id=False):
    knowledges = [get_value(ck, "content", "content_with_weight") for ck in kbinfos["chunks"]]
    kwlg_len = len(knowledges)
    used_token_count = 0
    chunks_num = 0

    for i, c in enumerate(knowledges):
        if not c:
            continue
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            # 超出 97% 预算：截断，后续 chunks 丢弃
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {len(knowledges)}/{kwlg_len}")
            break
```

**关键设计**：
- chunks 已经按相似度降序排列（检索阶段完成），所以截断的是**相关性最低**的 chunks
- 截断单位是完整的 chunk，不会把一个 chunk 切一半放进去（避免截断后语义不完整）
- 97% 而不是 100%，留 3% 给格式化开销（ID、Title、分隔符等）

### 3.2 每个 chunk 的格式化开销

```python
# 每个 chunk 的格式化模板（约 20-30 tokens 开销）
cnt = "\nID: {i}"
cnt += "\n├── Title: {document_name}"
cnt += "\n└── Content:\n{content}"
```

格式化开销是固定的，不会随内容增长，可以忽略不计。

---

## 四、第二层：消息列表的三级降级压缩

**代码位置**：`rag/prompts/generator.py` → `message_fit_in()` L66

这是整个长上下文管理的核心函数，实现了三级降级策略：

```python
def message_fit_in(msg, max_length=4000):
    def count():
        total = sum(num_tokens_from_string(m["content"]) for m in msg)
        return total

    # ── 第一级：总量检查 ──
    c = count()
    if c < max_length:
        return c, msg          # 未超限，原样返回

    # ── 第二级：丢弃中间历史，只保留系统 Prompt + 最新用户消息 ──
    msg_ = [m for m in msg if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])   # 只保留最后一条用户消息
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg          # 丢弃历史后满足，返回

    # ── 第三级：截断最大的那一方 ──
    ll  = num_tokens_from_string(msg_[0]["content"])   # 系统 Prompt 长度
    ll2 = num_tokens_from_string(msg_[-1]["content"])  # 用户消息长度

    if ll / (ll + ll2) > 0.8:
        # 系统 Prompt 占比 > 80%：截断系统 Prompt
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[0]["content"] = m
    else:
        # 用户消息更大：截断用户消息
        m = msg_[-1]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[-1]["content"] = m

    return max_length, msg
```

**三级降级逻辑**：

| 级别 | 触发条件 | 操作 | 保留内容 |
|------|---------|------|---------|
| 第一级 | 总 token < max_length | 不处理 | 全部 |
| 第二级 | 总 token ≥ max_length | 丢弃中间历史 | 系统 Prompt + 最新用户消息 |
| 第三级 | 丢弃历史后仍超限 | 截断最大方 | 系统 Prompt（截断）+ 用户消息 |

**设计逻辑**：
- 系统 Prompt（含知识块）是最重要的，优先保留
- 最新用户消息是必须的，不能丢
- 中间历史是可以牺牲的——因为查询改写（`full_question`）已经把历史信息融入了当前问题

---

## 五、第三层：对话历史的窗口截断

### 5.1 对话系统（Dialog）：取最近 N 条

**代码位置**：`api/db/services/dialog_service.py` L503-505，L661-662

```python
# 只取最近 3 条用户问题用于检索
questions = [m["content"] for m in messages if m["role"] == "user"][-3:]

# 所有历史消息传入 message_fit_in，由它决定保留多少
msg.extend([{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"])
used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
```

对话系统不做主动截断，而是把全部历史传给 `message_fit_in`，由它的三级降级机制处理。

### 5.2 Agent 系统：`message_history_window_size` 硬窗口

**代码位置**：`agent/canvas.py` → `get_history()` L726，`agent/component/base.py` L42

```python
# agent/component/base.py
class ComponentParamBase(ABC):
    def __init__(self):
        self.message_history_window_size = 13  # 默认保留 13 轮
```

```python
# agent/canvas.py
def get_history(self, window_size):
    convs = []
    if window_size <= 0:
        return convs
    # window_size * -2：每轮对话有 user + assistant 两条，取最近 N 轮
    for role, obj in self.history[window_size * -2:]:
        if isinstance(obj, dict):
            convs.append({"role": role, "content": obj.get("content", "")})
        else:
            convs.append({"role": role, "content": str(obj)})
    return convs
```

```python
# agent/component/llm.py → _prepare_prompt_variables()
msg, sys_prompt = self._sys_prompt_and_msg(
    self._canvas.get_history(self._param.message_history_window_size)[:-1],
    args
)
```

Agent 系统在**取历史时就截断**，而不是取完再压缩。`window_size * -2` 是因为每轮对话包含 user + assistant 两条消息，取最近 N 轮就是取最后 `N*2` 条。

取完历史后，还会再经过 `message_fit_in` 做最终的 Token 检查：

```python
# agent/component/llm.py → _invoke_async()
_, msg_fit = message_fit_in(
    [{"role": "system", "content": prompt}, *deepcopy(msg)],
    int(self.chat_mdl.max_length * 0.97),
)
```

**两层保障**：先用 `window_size` 控制历史条数，再用 `message_fit_in` 控制总 Token 数。

### 5.3 `form_history` 的内容截断

**代码位置**：`rag/prompts/generator.py` → `form_history()` L374

```python
def form_history(history, limit=-6):
    context = ""
    for h in history[limit:]:   # 默认只取最近 6 条
        if h["role"] == "system":
            continue
        role = "USER" if h["role"].upper() == "USER" else "AGENT"
        # 每条历史消息最多 2048 字符，超出截断并加 "..."
        context += f"\n{role}: {h['content'][:2048] + ('...' if len(h['content']) > 2048 else '')}"
    return context
```

用于 Agent 反思（`reflect_async`）等场景，对每条历史消息做**内容截断**（2048 字符），而不是丢弃整条。这样保留了所有轮次的摘要信息，但压缩了每条的详细内容。

---

## 六、第四层：查询改写消除历史依赖

**代码位置**：`rag/prompts/generator.py` → `full_question()` L228

```python
# api/db/services/dialog_service.py L545-547
if len(questions) > 1 and prompt_config.get("refine_multiturn"):
    questions = [await full_question(dialog.tenant_id, dialog.llm_id, messages)]
else:
    questions = questions[-1:]
```

`full_question` 把多轮对话中的指代词和省略信息补全到当前问题里：

```
历史：
  USER: 液压泵的额定压力是多少？
  ASSISTANT: 350bar
  USER: 那最大流量呢？

改写后：
  液压泵的最大流量是多少？
```

改写后的完整问题用于检索，**不再需要把历史对话传给检索引擎**。这从根本上减少了历史对话对 Token 预算的占用——历史信息已经被"蒸馏"进了当前问题。

---

## 七、第五层：Agent 的工具调用摘要记忆

Agent 在多轮工具调用场景下，每次工具调用的结果可能很长（如搜索返回 5000 字的网页内容）。RAGFlow 用摘要机制压缩这些结果。

### 7.1 工具调用结果摘要

**代码位置**：`rag/prompts/generator.py` → `tool_call_summary()` L458，`rag/prompts/summary4memory.md`

```python
async def tool_call_summary(chat_mdl, name: str, params: dict, result: str, ...) -> str:
    template = PROMPT_JINJA_ENV.from_string(SUMMARY4MEMORY)
    system_prompt = template.render(name=name, params=params, result=result)
    ans = await chat_mdl.async_chat(system_prompt, [{"role": "user", "content": "→ Summary: "}])
    return ans
```

`summary4memory.md` 的摘要规则：

```markdown
Rules:
1. Condense the response into 1-2 short sentences.
2. Never omit: Success/error status, Core results, Critical constraints
3. Exclude: timestamps, request IDs, verbose technical details

Template: "[Status] + [Key Outcome] + [Critical Constraints]"

Example:
  Tool Response: {"status": "success", "temperature": 78.2, "unit": "F", "location": "Tokyo", "timestamp": 16923456}
  → Summary: "Success: Tokyo temperature is 78°F."
```

原始工具结果可能有几千 token，摘要后压缩到 1-2 句话（约 20-50 token）。

### 7.2 记忆相关性排序

**代码位置**：`rag/prompts/generator.py` → `rank_memories_async()` L469，`rag/prompts/rank_memory.md`

当积累了多条工具调用摘要后，用 LLM 对它们按与当前目标的相关性排序：

```python
async def rank_memories_async(chat_mdl, goal, sub_goal, tool_call_summaries, ...):
    system_prompt = RANK_MEMORY.render(
        goal=goal,
        sub_goal=sub_goal,
        results=[{"i": i, "content": s} for i, s in enumerate(tool_call_summaries)]
    )
    ans = await chat_mdl.async_chat(system_prompt, [...], stop="<|stop|>")
    # 返回排序后的索引列表，如 [2, 0, 1]
```

排序后只把最相关的 Top-K 条摘要放入上下文，进一步压缩 Token 占用。

### 7.3 记忆存储到 Canvas

**代码位置**：`agent/canvas.py` L847-851

```python
def add_memory(self, user: str, assist: str, summ: str):
    self.memory.append((user, assist, summ))

def get_memory(self) -> list[Tuple]:
    return self.memory
```

摘要后的记忆存储在 Canvas 的 `memory` 列表中，持久化到 DSL JSON，跨轮次可用。

---

## 八、第六层：生成 Token 预算控制

**代码位置**：`api/db/services/dialog_service.py` L668-669

```python
if "max_tokens" in gen_conf:
    gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)
```

`used_token_count` 是 `message_fit_in` 返回的实际输入 Token 数。生成的最大 Token 数被限制为 `模型上下文窗口 - 实际输入 Token`，确保输入 + 输出不超过模型上下文窗口。

---

## 九、完整的 Token 预算分配流程

```
模型上下文窗口（如 8192 tokens）
    │
    ├─ Step 1: kb_prompt()
    │   按相似度排序的 chunks，累计到 97% 上限截断
    │   → 知识块占用：0 ~ 7946 tokens
    │
    ├─ Step 2: 系统 Prompt 组装
    │   system_prompt.format(knowledge=knowledges) + citation_prompt
    │   → 系统 Prompt 总量
    │
    ├─ Step 3: message_fit_in(msg, max_tokens * 0.95)
    │   三级降级：
    │   Level 1: 总量 < 7782 → 全部保留
    │   Level 2: 超限 → 丢弃中间历史，只保留 system + 最新 user
    │   Level 3: 仍超限 → 截断最大方（系统 Prompt 或用户消息）
    │   → used_token_count（实际输入 Token 数）
    │
    └─ Step 4: 生成预算
        gen_conf["max_tokens"] = min(用户设置, 8192 - used_token_count)
        → 确保输入 + 输出 ≤ 模型上下文窗口
```

---

## 十、哪些信息应保留，哪些可摘要或丢弃

基于 RAGFlow 的实现，总结信息优先级：

| 信息类型 | 策略 | 原因 |
|---------|------|------|
| 系统 Prompt（角色/规则） | 始终保留 | 决定模型行为的基础约束 |
| 召回知识块（高相似度） | 优先保留，按相似度截断 | 直接决定回答质量 |
| 最新用户问题 | 始终保留 | 当前任务的核心 |
| 引用指令 | 有知识块时保留 | 防幻觉的关键约束 |
| 近期对话历史（最近 N 轮） | 保留，超限时丢弃 | 多轮理解的上下文 |
| 远期对话历史 | 通过查询改写蒸馏后丢弃 | 已融入当前问题 |
| 工具调用原始结果 | 摘要后保留（1-2 句） | 原始结果冗余，摘要保留关键信息 |
| 召回知识块（低相似度） | 截断丢弃 | 相关性低，占用预算不值得 |
| 历史消息中的图片/附件 | 丢弃 | Token 消耗极大，历史图片通常不再需要 |

---

## 十一、代码位置速查表

| 功能 | 文件 | 函数/位置 |
|------|------|---------|
| max_tokens 确定 | `api/db/services/dialog_service.py` | L477 |
| 知识块 Token 截断 | `rag/prompts/generator.py` | `kb_prompt()` L103 |
| 消息列表三级降级 | `rag/prompts/generator.py` | `message_fit_in()` L66 |
| 生成 Token 预算控制 | `api/db/services/dialog_service.py` | L668–669 |
| Agent 历史窗口截断 | `agent/canvas.py` | `get_history()` L726 |
| Agent 默认窗口大小 | `agent/component/base.py` | L42（默认 13） |
| Agent 消息 Token 检查 | `agent/component/llm.py` | `_invoke_async()` L367 |
| 历史内容截断（2048字符） | `rag/prompts/generator.py` | `form_history()` L374 |
| 查询改写消除历史依赖 | `rag/prompts/generator.py` | `full_question()` L228 |
| 工具调用结果摘要 | `rag/prompts/generator.py` | `tool_call_summary()` L458 |
| 工具调用摘要 Prompt | `rag/prompts/summary4memory.md` | — |
| 记忆相关性排序 | `rag/prompts/generator.py` | `rank_memories_async()` L469 |
| 记忆排序 Prompt | `rag/prompts/rank_memory.md` | — |
| Canvas 记忆存储 | `agent/canvas.py` | `add_memory()` L847 |
