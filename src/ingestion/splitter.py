import re
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger
from .loader import Document


@dataclass
class TextChunk:
    content: str
    metadata: Dict
    chunk_id: str
    start_idx: int
    end_idx: int

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TextChunk(id={self.chunk_id}, chars={len(self.content)}, preview='{preview}')"

    def to_dict(self) -> Dict:
        return {"id": self.chunk_id, "text": self.content, "metadata": self.metadata}


class BaseSplitter(ABC):

    @abstractmethod
    def split(self, document: Document) -> List[TextChunk]:
        pass

    def split_batch(self, documents: List[Document]) -> List[TextChunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks


class RecursiveCharacterSplitter(BaseSplitter):
    """
    递归字符切分器（推荐默认）

    改进：
    - 策略一：从 _page_to_section 注入 section_path
    - 策略二：_merge_small_chunks 合并碎片块
    - 策略三：chunk_overlap 滑动窗口（已有）
    """

    def __init__(self, chunk_size=512, chunk_overlap=50, min_chunk_size=50,
                 separators=None, keep_separator=True, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.keep_separator = keep_separator
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        logger.info(f"RecursiveCharacterSplitter init (chunk_size={chunk_size}, overlap={chunk_overlap}, min={min_chunk_size})")

    def split(self, document: Document) -> List[TextChunk]:
        raw_chunks = self._recursive_split(document.content, self.separators[:])
        text_chunks = []
        current_pos = 0
        current_page = 1

        # 策略一：页码 → 标题路径映射（由 data_loader 写入）
        page_to_section = document.metadata.get("_page_to_section", {})
        # ChromaDB 只支持标量类型，过滤掉嵌套 dict
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

        for i, content in enumerate(raw_chunks):
            start_idx = document.content.find(content, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(content)

            page_match = re.search(r'\[PAGE:(\d+)\]', content)
            if page_match:
                current_page = int(page_match.group(1))

            clean_content = re.sub(r'\[PAGE:\d+\]\s*', '', content).strip()
            section_path = page_to_section.get(current_page - 1, '')

            metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "chunk_size": len(clean_content),
                "parent_doc_id": document.doc_id,
                "page": current_page,
                "section_path": section_path,
            }

            text_chunks.append(TextChunk(
                content=clean_content,
                metadata=metadata,
                chunk_id=f"{document.doc_id}_chunk_{i}",
                start_idx=start_idx,
                end_idx=end_idx,
            ))
            current_pos = end_idx

        # 策略二：合并碎片块
        return self._merge_small_chunks(text_chunks)

    def split_hierarchical(
        self,
        document: Document,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
    ) -> tuple:
        """
        父子分块：小块用于精确检索，大块用于喂给 LLM 补充上下文。

        流程：
        1. 用较大的 chunk_size 切出父块（parent chunks）
        2. 对每个父块再用当前 chunk_size 切出子块（child chunks）
        3. 子块 metadata 记录 parent_chunk_id，检索命中子块后可取出父块

        Args:
            document: 待切分文档
            parent_chunk_size: 父块大小，默认 1024 字符
            parent_chunk_overlap: 父块重叠，默认 100 字符

        Returns:
            (child_chunks, parent_chunks)
            - child_chunks: 入向量库，用于检索
            - parent_chunks: 存储备用，检索命中后取出喂给 LLM
        """
        page_to_section = document.metadata.get("_page_to_section", {})
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

        # ---- 第一步：切父块 ----
        parent_splitter = RecursiveCharacterSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            separators=self.separators,
        )
        parent_raw = parent_splitter._recursive_split(document.content, self.separators[:])

        parent_chunks: List[TextChunk] = []
        current_pos = 0
        current_page = 1

        for i, content in enumerate(parent_raw):
            start_idx = document.content.find(content, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(content)

            page_match = re.search(r'\[PAGE:(\d+)\]', content)
            if page_match:
                current_page = int(page_match.group(1))

            clean_content = re.sub(r'\[PAGE:\d+\]\s*', '', content).strip()
            section_path = page_to_section.get(current_page - 1, '')
            parent_id = f"{document.doc_id}_parent_{i}"

            parent_chunks.append(TextChunk(
                content=clean_content,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "chunk_size": len(clean_content),
                    "parent_doc_id": document.doc_id,
                    "page": current_page,
                    "section_path": section_path,
                    "is_parent": True,
                },
                chunk_id=parent_id,
                start_idx=start_idx,
                end_idx=end_idx,
            ))
            current_pos = end_idx

        # ---- 第二步：对每个父块切子块 ----
        child_chunks: List[TextChunk] = []
        child_idx = 0

        for parent in parent_chunks:
            child_raw = self._recursive_split(parent.content, self.separators[:])
            for raw_content in child_raw:
                clean_content = raw_content.strip()
                if len(clean_content) < self.min_chunk_size:
                    continue
                child_chunks.append(TextChunk(
                    content=clean_content,
                    metadata={
                        **parent.metadata,
                        "chunk_index": child_idx,
                        "chunk_size": len(clean_content),
                        "parent_chunk_id": parent.chunk_id,  # 关键：记录父块 ID
                        "is_parent": False,
                    },
                    chunk_id=f"{document.doc_id}_child_{child_idx}",
                    start_idx=parent.start_idx,
                    end_idx=parent.end_idx,
                ))
                child_idx += 1

        logger.info(
            f"Hierarchical split: {len(parent_chunks)} parent chunks, "
            f"{len(child_chunks)} child chunks"
        )
        return child_chunks, parent_chunks

    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """把长度小于 min_chunk_size 的碎片块合并到前一个块。"""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk.content) < self.min_chunk_size and merged:
                prev = merged[-1]
                merged[-1] = TextChunk(
                    content=prev.content + "\n" + chunk.content,
                    metadata={**prev.metadata, "chunk_size": len(prev.content) + len(chunk.content) + 1},
                    chunk_id=prev.chunk_id,
                    start_idx=prev.start_idx,
                    end_idx=chunk.end_idx,
                )
            else:
                merged.append(chunk)
        return merged

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        separator = separators[0] if separators else ""
        remaining = separators[1:] if len(separators) > 1 else []

        if not separator:
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
            return [c for c in chunks if c.strip()]

        parts = text.split(separator)
        chunks = []
        current = ""

        for part in parts:
            piece = part + (separator if self.keep_separator else "")
            if self.length_function(current + piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    if self.length_function(current) > self.chunk_size and remaining:
                        chunks.extend(self._recursive_split(current, remaining))
                    else:
                        chunks.append(current)
                current = piece

        if current.strip():
            if self.length_function(current) > self.chunk_size and remaining:
                chunks.extend(self._recursive_split(current, remaining))
            else:
                chunks.append(current)

        merged = []
        buffer = ""
        for chunk in chunks:
            if self.length_function(buffer + chunk) <= self.chunk_size:
                buffer += chunk
            else:
                if buffer.strip():
                    merged.append(buffer)
                buffer = chunk
        if buffer.strip():
            merged.append(buffer)

        return merged if merged else [text]


class SemanticSplitter(BaseSplitter):

    def __init__(self, embedding_model=None, max_chunk_size=512, similarity_threshold=0.8):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        if embedding_model is None:
            logger.warning("No embedding model provided, fallback to sentence splitting")

    def split(self, document: Document) -> List[TextChunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []
        chunks = []
        current_chunk = [sentences[0]]
        for sentence in sentences[1:]:
            if len(" ".join(current_chunk)) + len(sentence) > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return self._create_chunks(document, chunks)

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks(self, document: Document, chunks: List[str]) -> List[TextChunk]:
        result = []
        current_pos = 0
        for i, content in enumerate(chunks):
            start_idx = document.content.find(content, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(content)
            result.append(TextChunk(
                content=content,
                metadata={**document.metadata, "chunk_index": i, "parent_doc_id": document.doc_id},
                chunk_id=f"{document.doc_id}_chunk_{i}",
                start_idx=start_idx,
                end_idx=end_idx,
            ))
            current_pos = end_idx
        return result


class FixedSizeSplitter(BaseSplitter):

    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata={**document.metadata, "chunk_index": idx, "parent_doc_id": document.doc_id},
                    chunk_id=f"{document.doc_id}_chunk_{idx}",
                    start_idx=start,
                    end_idx=end,
                ))
                idx += 1
            start = end - self.chunk_overlap
            if start >= end:
                break
        return chunks


class SplitterFactory:

    @staticmethod
    def create(splitter_type: str, **kwargs) -> BaseSplitter:
        splitters = {
            "recursive": RecursiveCharacterSplitter,
            "semantic": SemanticSplitter,
            "fixed": FixedSizeSplitter,
        }
        cls = splitters.get(splitter_type.lower())
        if not cls:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
        return cls(**kwargs)
