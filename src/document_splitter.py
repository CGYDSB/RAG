"""
Document Splitting and Chunking Strategies

支持策略：
1. 递归切分 RecursiveCharacterSplitter（推荐默认）
2. 语义切分 SemanticSplitter
3. 固定长度切分 FixedSizeSplitter
"""

import re
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from loguru import logger
from .data_loader import Document


# =========================================
# 1. 文本块结构
# =========================================
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
        return {
            "id": self.chunk_id,
            "text": self.content,
            "metadata": self.metadata,
        }


# =========================================
# 2. 抽象基类
# =========================================
class BaseSplitter(ABC):

    @abstractmethod
    def split(self, document: Document) -> List[TextChunk]:
        pass

    def split_batch(self, documents: List[Document]) -> List[TextChunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks


# =========================================
# 3. 递归切分（推荐）
# =========================================
class RecursiveCharacterSplitter(BaseSplitter):

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        length_function: Callable[[str], int] = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        logger.info(f"RecursiveCharacterSplitter init (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def split(self, document: Document) -> List[TextChunk]:
        raw_chunks = self._recursive_split(document.content, self.separators[:])
        text_chunks = []
        current_pos = 0
        current_page = 1

        for i, content in enumerate(raw_chunks):
            start_idx = document.content.find(content, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(content)

            # 从chunk内容中提取页码标记
            import re
            page_match = re.search(r'\[PAGE:(\d+)\]', content)
            if page_match:
                current_page = int(page_match.group(1))
            # 清理页码标记，不存入向量
            clean_content = re.sub(r'\[PAGE:\d+\]\s*', '', content).strip()

            metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "chunk_size": len(clean_content),
                "parent_doc_id": document.doc_id,
                "page": current_page,
            }

            text_chunks.append(TextChunk(
                content=clean_content,
                metadata=metadata,
                chunk_id=f"{document.doc_id}_chunk_{i}",
                start_idx=start_idx,
                end_idx=end_idx,
            ))
            current_pos = end_idx

        return text_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        separator = separators[0] if separators else ""
        remaining = separators[1:] if len(separators) > 1 else []

        if not separator:
            # 字符级切分
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

        # 合并过小的块
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


# =========================================
# 4. 语义切分
# =========================================
class SemanticSplitter(BaseSplitter):

    def __init__(
        self,
        embedding_model=None,
        max_chunk_size: int = 512,
        similarity_threshold: float = 0.8
    ):
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


# =========================================
# 5. 固定长度切分
# =========================================
class FixedSizeSplitter(BaseSplitter):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
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
                    metadata={
                        **document.metadata,
                        "chunk_index": idx,
                        "parent_doc_id": document.doc_id,
                    },
                    chunk_id=f"{document.doc_id}_chunk_{idx}",
                    start_idx=start,
                    end_idx=end,
                ))
                idx += 1

            start = end - self.chunk_overlap
            if start >= end:
                break

        return chunks


# =========================================
# 6. 工厂
# =========================================
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
