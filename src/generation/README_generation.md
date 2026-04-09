# generation 模块

负责 LLM 调用、Prompt 模板管理和上下文构建，是 RAG 系统的生成层。

---

## 文件结构

```
generation/
├── __init__.py     # 公共接口导出
└── generator.py    # LLM 生成器 + Prompt 管理
```

---

## generator.py

| 类 | 说明 |
|----|------|
| `GenerationConfig` | 生成参数封装（temperature、max_tokens、top_p 等） |
| `GenerationResult` | 生成结果封装（text、token 统计、finish_reason） |
| `BaseGenerator` | 抽象基类，定义 `generate` / `generate_async` / `generate_stream` 接口 |
| `OpenAIGenerator` | OpenAI / DeepSeek 兼容接口，支持同步、异步、流式输出 |
| `AzureOpenAIGenerator` | Azure OpenAI 封装 |
| `GeneratorFactory` | 工厂类，根据 `settings.yaml` 的 `llm.provider` 创建 |
| `PromptTemplate` | 单个 Prompt 模板，支持 `{variable}` 动态填充 |
| `PromptManager` | 从 `config/prompts.yaml` 加载多套模板，按场景切换 |
| `ContextBuilder` | 将检索结果拼接成符合 token 限制的上下文 |

**Prompt 模板体系**（`config/prompts.yaml`）：

| 模板 key | 场景 |
|----------|------|
| `system.rag_assistant` | 默认问答，9 条防幻觉规则 |
| `system.conversational` | 多轮对话，携带历史摘要 |
| `system.concise` | 快速返回，简短回答 |
| `system.detailed` | 报告场景，结构化输出 |

**配置示例**：

```yaml
llm:
  provider: "openai"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
  temperature: 0.1
  max_tokens: 1000
```

---

## 公共接口

```python
from src.generation import (
    GeneratorFactory,       # 创建 LLM 生成器
    OpenAIGenerator,        # 直接使用 OpenAI 兼容接口
    PromptManager,          # Prompt 模板管理
    ContextBuilder,         # 上下文构建
    GenerationConfig,       # 生成参数
    GenerationResult,       # 生成结果
)
```
