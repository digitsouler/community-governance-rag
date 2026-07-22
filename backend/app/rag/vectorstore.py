"""向量库封装。

默认：纯 Python 内存向量库（cosine 相似度，零依赖，保证任意环境可跑）。
可选升级：配置 qdrant_url 且环境已安装 qdrant-client 时，自动切换为 Qdrant
服务端（生产推荐，支持大规模与持久化）。

对外接口统一：upsert / search / count / reset。
"""
from __future__ import annotations

import math
from typing import Any

from app.config import Settings, get_settings
from app.log import get_logger

log = get_logger("rag.vectorstore")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class PureVectorStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.name = settings.collection_name
        self._data: list[dict] = []

    def _ensure(self):
        return  # 内存结构无需预建

    def upsert(self, points: list[dict]):
        # points: [{id, vector, payload}]
        self._data = [p for p in self._data if p["id"] not in {x["id"] for x in points}]
        self._data.extend(points)

    def count(self) -> int:
        return len(self._data)

    def search(self, vector: list[float], top_k: int, category: str | None = None) -> list[dict]:
        cands = self._data
        if category:
            cands = [d for d in cands if d["payload"].get("category") == category]
        scored = [
            {"id": d["id"], "score": _cosine(vector, d["vector"]), "payload": d["payload"]}
            for d in cands
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        if not scored:
            log.warning("向量检索返回 0 条（库可能为空）")
        return scored[:top_k]

    def reset(self):
        self._data = []


def _build_qdrant_store(settings: Settings):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    class QdrantStore:
        def __init__(self):
            if settings.qdrant_url:
                self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            else:
                self.client = QdrantClient(":memory:")
            self.name = settings.collection_name
            if not self.client.collection_exists(self.name):
                self.client.create_collection(
                    collection_name=self.name,
                    vectors_config=VectorParams(size=settings.vector_dim, distance=Distance.COSINE),
                )

        def upsert(self, points: list[dict]):
            if not points:
                return
            self.client.upsert(
                collection_name=self.name,
                points=[PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points],
            )

        def count(self) -> int:
            return self.client.count(self.name).count

        def search(self, vector: list[float], top_k: int, category: str | None = None):
            qf = None
            if category:
                qf = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
            hits = self.client.query_points(
                collection_name=self.name, query=vector, limit=top_k, query_filter=qf, with_payload=True
            ).points
            return [{"id": h.id, "score": float(h.score), "payload": h.payload or {}} for h in hits]

        def reset(self):
            if self.client.collection_exists(self.name):
                self.client.delete_collection(self.name)
            self.__init__()

    return QdrantStore()


class VectorStore:
    """外观类：根据环境与配置选择底层实现。"""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._impl = None
        if self.s.qdrant_url:
            try:
                self._impl = _build_qdrant_store(self.s)
            except Exception as e:
                print(f"[向量库] Qdrant 初始化失败，回退纯内存：{e}")
        if self._impl is None:
            self._impl = PureVectorStore(self.s)

    def upsert(self, points: list[dict]):
        self._impl.upsert(points)

    def count(self) -> int:
        return self._impl.count()

    def search(self, vector: list[float], top_k: int, category: str | None = None):
        return self._impl.search(vector, top_k, category)

    def reset(self):
        self._impl.reset()


_store_instance: "VectorStore | None" = None


def get_vector_store(settings: Settings | None = None) -> "VectorStore":
    """全局单例：保证入库与查询共用同一份内存索引（或同一 Qdrant 连接）。"""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(settings or get_settings())
    return _store_instance
