"""
DataLoader — 文档加载模块

支持格式：PDF / TXT / MD / DOCX / HTML
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


# =========================================
# 文档数据结构
# =========================================
@dataclass
class Document:
    """
    标准文档结构

    Attributes:
        content:  文档正文
        metadata: 元信息（来源、文件名、页码等）
        doc_id:   唯一标识
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Document(id={self.doc_id[:8]}, chars={len(self.content)}, preview='{preview}')"


# =========================================
# 文档加载器
# =========================================
class DataLoader:
    """
    多格式文档加载器

    用法：
        loader = DataLoader()
        doc  = loader.load_document("report.pdf")
        docs = loader.load_directory("data/raw/")
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_file_size_mb = self.config.get("max_file_size", 50)

    # --------------------------------------------------
    # 单文件加载（根据扩展名分发）
    # --------------------------------------------------
    def load_document(self, path) -> Document:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max {self.max_file_size_mb}MB)")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}")

        loaders = {
            ".pdf":  self._load_pdf,
            ".txt":  self._load_text,
            ".md":   self._load_text,
            ".docx": self._load_docx,
            ".html": self._load_html,
            ".htm":  self._load_html,
        }

        content, metadata = loaders[ext](path)

        metadata.update({
            "source": str(path),
            "filename": path.name,
            "extension": ext,
            "file_size_mb": round(size_mb, 3),
        })

        logger.info(f"Loaded: {path.name} ({len(content)} chars)")
        return Document(content=content, metadata=metadata)

    # --------------------------------------------------
    # 目录批量加载
    # --------------------------------------------------
    def load_directory(self, directory, recursive: bool = True) -> List[Document]:
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        logger.info(f"Found {len(files)} files in {directory}")

        documents = []
        for f in files:
            try:
                documents.append(self.load_document(f))
            except Exception as e:
                logger.warning(f"Skipping {f.name}: {e}")

        logger.info(f"Loaded {len(documents)}/{len(files)} documents")
        return documents

    # --------------------------------------------------
    # PDF 加载
    # --------------------------------------------------
    def _load_pdf(self, path: Path):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf")

        reader = PdfReader(str(path))
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                text = self._clean_pdf_text(text)
                # 在每页开头插入页码标记，切分后 chunk 可以反查
                pages.append(f"[PAGE:{i+1}] {text}")

        content = "\n\n".join(pages)
        metadata = {"page_count": len(reader.pages), "type": "pdf"}
        return content, metadata

    def _clean_pdf_text(self, text: str) -> str:
        """清洗PDF提取的文本，去除多余空白和断行"""
        import re
        # 把单个换行（断词）替换为空格，保留双换行（段落）
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # 去除多余空格和制表符
        text = re.sub(r'[ \t]+', ' ', text)
        # 还原段落分隔
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 关键：去除中文字符之间的空格（PDF提取常见问题）
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        # 去除中文和标点之间的空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[，。！？、；：""''（）【】])', '', text)
        text = re.sub(r'(?<=[，。！？、；：""''（）【】])\s+(?=[\u4e00-\u9fff])', '', text)
        return text.strip()

    # --------------------------------------------------
    # TXT / MD 加载
    # --------------------------------------------------
    def _load_text(self, path: Path):
        encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]

        for enc in encodings:
            try:
                content = path.read_text(encoding=enc)
                return content, {"type": path.suffix.lstrip(".")}
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Cannot decode file: {path}")

    # --------------------------------------------------
    # DOCX 加载
    # --------------------------------------------------
    def _load_docx(self, path: Path):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(paragraphs)
        metadata = {"paragraph_count": len(paragraphs), "type": "docx"}
        return content, metadata

    # --------------------------------------------------
    # HTML 加载
    # --------------------------------------------------
    def _load_html(self, path: Path):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # 移除 script / style 标签
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        content = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else ""
        metadata = {"title": title, "type": "html"}
        return content, metadata
