# ingestion 模块

负责文档加载、解析和切分，是 RAG 系统的数据入口层。

---

## 文件结构

```
ingestion/
├── __init__.py     # 公共接口导出
├── loader.py       # 多格式文档加载
└── splitter.py     # 文档切分策略
```

---

## loader.py — 文档加载

支持 PDF / TXT / MD / DOCX / HTML 五种格式，PDF 处理有专项优化。

| 类 | 说明 |
|----|------|
| `Document` | 标准文档结构，包含 `content`、`metadata`、`doc_id` |
| `DataLoader` | 多格式文档加载器，支持单文件和目录批量加载 |

**PDF 专项优化**：

- `_clean_pdf_text()`：清洗中文 PDF 常见乱码（去除中文字符间多余空格、断行修复）
- `_extract_outline()`：提取 PDF 书签目录，建立 0-based 页码 → 章节标题映射（策略一）
- `_build_page_to_section()`：按页码区间建立映射表，传递给切分器注入 `section_path`
- `_detect_toc_pages()`：识别目录页并过滤，避免目录条目干扰检索（策略五）

每页文本插入 `[PAGE:N]` 标记，切分后 chunk 可反查原始页码。

---

## splitter.py — 文档切分

| 类 | 说明 |
|----|------|
| `TextChunk` | 文本块结构，包含 `content`、`metadata`（含 `section_path`、`page`）、`chunk_id` |
| `RecursiveCharacterSplitter` | 递归字符切分（推荐默认），按语义边界优先级切分 |
| `SemanticSplitter` | 语义切分，基于句子边界 |
| `FixedSizeSplitter` | 固定长度切分 |
| `SplitterFactory` | 工厂类，根据类型字符串创建切分器 |

**RecursiveCharacterSplitter 改进**：

- 策略一：从 `_page_to_section` 注入 `section_path`，每个 chunk 携带章节归属
- 策略二：`_merge_small_chunks()`，合并长度 < `min_chunk_size` 的碎片块
- 策略三：`chunk_overlap` 滑动窗口，保留边界处语义连续性
- 父子分块：`split_hierarchical()`，小块（子 chunk）用于精确检索，大块（父 chunk）喂给 LLM

**关键参数**（`settings.yaml`）：

```yaml
document:
  default_chunk_size: 256
  default_chunk_overlap: 30
  splitting:
    hierarchical:
      enabled: false
      parent_chunk_size: 1024
      parent_chunk_overlap: 100
```

---

## 公共接口

```python
from src.ingestion import (
    DataLoader,                     # 文档加载
    Document,                       # 文档数据结构
    TextChunk,                      # 文本块数据结构
    RecursiveCharacterSplitter,     # 递归切分器
    SplitterFactory,                # 切分器工厂
)
```
