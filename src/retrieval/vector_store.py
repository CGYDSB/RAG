"""
Vector Store Management

支持后端：
- ChromaDB（本地，开发推荐）
- Qdrant（生产推荐）
- Redis（缓存型）
"""

import uuid
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import numpy as np


# =========================================
# 1. 向量记录结构
# =========================================
@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata,
        }


# =========================================
# 2. 抽象基类
# =========================================
class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, records: List[VectorRecord]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def get_by_id(self, id: str) -> Optional[VectorRecord]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


# =========================================
# 3. ChromaDB（本地，开发推荐）
# =========================================
class ChromaVectorStore(BaseVectorStore):

    def __init__(self, collection_name: str = "documents",
                 persist_directory: str = "./data/chroma_db"):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaVectorStore ready: {collection_name} @ {persist_directory}")

    def add(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        self.collection.add(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata for r in records],
        )

    def search(self, query_vector: List[float], top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict]:
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count() or 1),
        )
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": 1 - result["distances"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def get_by_id(self, id: str) -> Optional[VectorRecord]:
        result = self.collection.get(ids=[id])
        if not result["ids"]:
            return None
        return VectorRecord(
            id=result["ids"][0],
            vector=[],
            text=result["documents"][0],
            metadata=result["metadatas"][0],
        )

    def get_by_ids(self, ids: List[str]) -> List[VectorRecord]:
        """批量获取多个记录，用于父子 chunk 召回时取出父块内容。"""
        if not ids:
            return []
        result = self.collection.get(ids=ids)
        records = []
        for i in range(len(result["ids"])):
            records.append(VectorRecord(
                id=result["ids"][i],
                vector=[],
                text=result["documents"][i],
                metadata=result["metadatas"][i],
            ))
        return records

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name)


# =========================================
# 4. Qdrant（生产推荐）
# =========================================
class QdrantVectorStore(BaseVectorStore):

    def __init__(self, collection_name: str = "documents", vector_size: int = 384,
                 host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        logger.info(f"QdrantVectorStore ready: {collection_name}")

    def add(self, records: List[VectorRecord]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=r.id,
                vector=r.vector,
                payload={"text": r.text, **r.metadata},
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {
                "id": str(r.id),
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
                "score": r.score,
            }
            for r in results
        ]

    def delete(self, ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def get_by_id(self, id: str) -> Optional[VectorRecord]:
        results = self.client.retrieve(collection_name=self.collection_name, ids=[id])
        if not results:
            return None
        r = results[0]
        return VectorRecord(
            id=str(r.id),
            vector=[],
            text=r.payload.get("text", ""),
            metadata={k: v for k, v in r.payload.items() if k != "text"},
        )

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)


# =========================================
# 5. Redis（缓存型）
# =========================================
class RedisVectorStore(BaseVectorStore):

    def __init__(self, host: str = "localhost", port: int = 6379,
                 password: Optional[str] = None):
        import redis
        self.client = redis.Redis(host=host, port=port, password=password)
        self._ids: List[str] = []

    def add(self, records: List[VectorRecord]) -> None:
        for r in records:
            self.client.hset(r.id, mapping={
                "text": r.text,
                "vector": np.array(r.vector, dtype=np.float32).tobytes(),
            })
            if r.id not in self._ids:
                self._ids.append(r.id)

    def search(self, query_vector: List[float], top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict]:
        q = np.array(query_vector, dtype=np.float32)
        scores = []
        for rid in self._ids:
            raw = self.client.hget(rid, "vector")
            if raw:
                v = np.frombuffer(raw, dtype=np.float32)
                norm = np.linalg.norm(q) * np.linalg.norm(v)
                score = float(np.dot(q, v) / norm) if norm > 0 else 0.0
                scores.append((rid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rid, score in scores[:top_k]:
            text = (self.client.hget(rid, "text") or b"").decode()
            results.append({"id": rid, "text": text, "metadata": {}, "score": score})
        return results

    def delete(self, ids: List[str]) -> None:
        self.client.delete(*ids)
        self._ids = [i for i in self._ids if i not in ids]

    def get_by_id(self, id: str) -> Optional[VectorRecord]:
        data = self.client.hgetall(id)
        if not data:
            return None
        return VectorRecord(id=id, vector=[], text=data.get(b"text", b"").decode(), metadata={})

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        if self._ids:
            self.client.delete(*self._ids)
        self._ids = []


# =========================================
# 6. 工厂
# =========================================
class VectorStoreFactory:

    @staticmethod
    def create(config: Dict) -> BaseVectorStore:
        provider = config.get("provider", "chroma")

        if provider == "chroma":
            cfg = config.get("chroma", {})
            return ChromaVectorStore(
                collection_name=cfg.get("collection_name", "documents"),
                persist_directory=cfg.get("persist_directory", "./data/chroma_db"),
            )
        elif provider == "qdrant":
            cfg = config.get("qdrant", {})
            return QdrantVectorStore(
                collection_name=cfg.get("collection_name", "documents"),
                vector_size=cfg.get("vector_size", 384),
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 6333),
            )
        elif provider == "redis":
            cfg = config.get("redis", {})
            return RedisVectorStore(
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 6379),
                password=cfg.get("password"),
            )
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")
