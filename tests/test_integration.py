"""
这个 test_integration.py 是 RAG 系统的集成测试模块，
用于验证从文档加载、文本切分、向量化、向量存储到完整 RAG 流水线的全流程是否正常工作，
通过 pytest 构建单元测试与集成测试用例，结合临时文件与临时数据库环境，
确保各模块接口兼容性与数据流正确性，从而提高系统的稳定性与生产可靠性。
====================================================
📌 RAG系统集成测试文件（Integration Test）
====================================================

该文件用于测试整个RAG系统的核心模块是否能协同工作，包括：

1️⃣ 文档加载模块（DataLoader）
2️⃣ 文本切分模块（Splitter）
3️⃣ 向量化模型（Embedding）
4️⃣ 向量数据库（Vector Store）
5️⃣ 完整RAG流水线（Pipeline）

测试目标：
- 验证各模块功能正确性
- 验证模块之间接口是否兼容
- 验证基础数据流是否正确
"""

# =========================
# 📌 基础依赖
# =========================
import pytest
import tempfile
import os
from pathlib import Path

# =========================
# 📌 被测试的核心模块
# =========================
from src.ingestion.loader import DataLoader, Document
from src.ingestion.splitter import RecursiveCharacterSplitter
from src.embedding.model import SentenceTransformerEmbedding
from src.retrieval.vector_store import ChromaVectorStore
from src.pipeline import RAGPipeline


# ====================================================
# 📌 1. 测试：文档加载模块
# ====================================================
class TestDataLoader:
    """测试 DataLoader 文档加载功能"""

    # -------------------------
    # ✔ 正常文本文件加载测试
    # -------------------------
    def test_load_text_file(self):
        # 创建一个临时txt文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            temp_path = f.name

        try:
            loader = DataLoader()

            # 加载文档
            doc = loader.load_document(temp_path)

            # =========================
            # 📌 断言验证
            # =========================
            assert isinstance(doc, Document)  # 返回必须是Document对象
            assert doc.content == "This is a test document."  # 内容正确
            assert doc.metadata['extension'] == '.txt'  # 元数据正确

        finally:
            # 删除临时文件
            os.remove(temp_path)

    # -------------------------
    # ❌ 不支持格式测试
    # -------------------------
    def test_unsupported_format(self):
        # 创建一个未知格式文件
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name

        try:
            loader = DataLoader()

            # 应该抛出 ValueError
            with pytest.raises(ValueError):
                loader.load_document(temp_path)

        finally:
            os.remove(temp_path)


# ====================================================
# 📌 2. 测试：文本切分模块
# ====================================================
class TestDocumentSplitter:
    """测试文本切分逻辑"""

    def test_recursive_split(self):
        # 初始化切分器（控制chunk大小）
        splitter = RecursiveCharacterSplitter(
            chunk_size=100,
            chunk_overlap=20
        )

        # 构造测试文档
        doc = Document(
            content="This is a test. " * 50,
            metadata={"source": "test"}
        )

        # 执行切分
        chunks = splitter.split(doc)

        # =========================
        # 📌 断言
        # =========================
        assert len(chunks) > 1  # 必须被切分
        assert all(len(c.content) <= 100 for c in chunks)  # 每块不超过限制
        assert all(hasattr(c, 'chunk_id') for c in chunks)  # 必须有ID


# ====================================================
# 📌 3. 测试：Embedding模型
# ====================================================
class TestEmbeddingModel:
    """测试向量化模型"""

    def test_sentence_transformer(self):
        # 初始化embedding模型（CPU模式）
        model = SentenceTransformerEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        texts = ["Hello world", "Test document"]

        # 生成向量
        embeddings = model.embed_documents(texts)

        # =========================
        # 📌 断言
        # =========================
        assert len(embeddings) == 2  # 两个文本 → 两个向量
        assert len(embeddings[0]) == model.dimension  # 向量维度正确
        assert model.dimension == 384  # MiniLM固定维度


# ====================================================
# 📌 4. 测试：向量数据库
# ====================================================
class TestVectorStore:
    """测试向量存储与检索"""

    def test_chroma_add_and_search(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:

            store = ChromaVectorStore(
                collection_name="test",
                persist_directory=tmpdir
            )

            from src.retrieval.vector_store import VectorRecord

            records = [
                VectorRecord(
                    id="1",
                    vector=[0.1] * 384,
                    text="Test document 1",
                    metadata={"source": "test"}
                ),
                VectorRecord(
                    id="2",
                    vector=[0.2] * 384,
                    text="Test document 2",
                    metadata={"source": "test"}
                )
            ]

            store.add(records)
            assert store.count() == 2

            results = store.search([0.1] * 384, top_k=1)
            assert len(results) == 1
            assert results[0]['id'] == "1"

            # 主动释放连接，避免 Windows 文件锁
            del store


# ====================================================
# 📌 5. 测试：完整RAG流水线
# ====================================================
class TestRAGPipeline:
    """测试完整RAG系统初始化"""

    def test_pipeline_initialization(self):
        # 配置模拟（不依赖真实API）
        config = {
            'embedding': {
                'provider': 'sentence-transformers',
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'cpu'
            },
            'vector_store': {
                'provider': 'chroma',
                'chroma': {'persist_directory': None}
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo'
            }
        }

        # 注意：
        # 没有API key时会失败，但用于验证初始化流程是否正确
        with pytest.raises(Exception):
            pipeline = RAGPipeline(config)


# ====================================================
# 📌 本地运行入口
# ====================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])