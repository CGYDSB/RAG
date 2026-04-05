"""
Embedding Model Management

支持：
1. OpenAI Embedding（API调用）
2. SentenceTransformers（本地，推荐CPU）
3. HuggingFace（自定义模型）
"""

import os
import hashlib
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np
from loguru import logger


# =========================================
# 1. 抽象基类
# =========================================
class BaseEmbeddingModel(ABC):

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


# =========================================
# 2. OpenAI Embedding
# =========================================
class OpenAIEmbedding(BaseEmbeddingModel):

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        normalize: bool = True
    ):
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)
        self._dimension = 1536
        logger.info(f"OpenAIEmbedding initialized: {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings = [item.embedding for item in response.data]
            if self.normalize:
                embeddings = [self._normalize(e) for e in embeddings]
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _normalize(self, emb: List[float]) -> List[float]:
        arr = np.array(emb)
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm != 0 else emb

    @property
    def dimension(self) -> int:
        return self._dimension


# =========================================
# 3. SentenceTransformers（本地，推荐CPU）
# =========================================
class SentenceTransformerEmbedding(BaseEmbeddingModel):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        # bge 系列模型 query 需要加前缀
        self._query_prefix = "为这个句子生成表示以用于检索相关文章：" if "bge" in model_name.lower() else ""
        logger.info(f"SentenceTransformerEmbedding ready, dim={self._dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._query_prefix:
            text = self._query_prefix + text
        return self.model.encode(text, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


# =========================================
# 4. HuggingFace Embedding
# =========================================
class HuggingFaceEmbedding(BaseEmbeddingModel):

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._dimension = self.model.config.hidden_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import torch

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# =========================================
# 5. LRU 缓存
# =========================================
class EmbeddingCache:

    def __init__(self, max_size: int = 10000):
        self.cache: Dict = {}
        self.order: List[str] = []
        self.max_size = max_size

    def _key(self, text: str, model: str) -> str:
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()

    def get(self, text: str, model: str):
        return self.cache.get(self._key(text, model))

    def set(self, text: str, model: str, emb) -> None:
        key = self._key(text, model)
        if len(self.cache) >= self.max_size:
            oldest = self.order.pop(0)
            self.cache.pop(oldest, None)
        self.cache[key] = emb
        self.order.append(key)


# =========================================
# 6. 带缓存的装饰器
# =========================================
class CachedEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, base_model: BaseEmbeddingModel):
        self.base_model = base_model
        self.cache = EmbeddingCache()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            cached = self.cache.get(text, "model")
            if cached is not None:
                results.append(cached)
            else:
                emb = self.base_model.embed_query(text)
                self.cache.set(text, "model", emb)
                results.append(emb)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self.base_model.dimension


# =========================================
# 7. 工厂
# =========================================
class EmbeddingModelFactory:

    @staticmethod
    def create(config: Dict) -> BaseEmbeddingModel:
        provider = config.get("provider", "sentence-transformers")

        if provider == "openai":
            model = OpenAIEmbedding(
                model=config.get("model", "text-embedding-ada-002"),
                batch_size=config.get("batch_size", 100),
                normalize=config.get("normalize_embeddings", True)
            )
        elif provider == "sentence-transformers":
            model = SentenceTransformerEmbedding(
                model_name=config.get("model", "all-MiniLM-L6-v2"),
                device=config.get("device", "cpu")
            )
        elif provider == "huggingface":
            model = HuggingFaceEmbedding(model_name=config.get("model"))
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

        if config.get("cache_enabled", True):
            model = CachedEmbeddingModel(model)

        return model
