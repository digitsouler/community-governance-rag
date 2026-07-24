"""向量库封装。

默认：纯 Python 内存向量库（cosine 相似度，零依赖，保证任意环境可跑）。
可选升级：配置 qdrant_url 且环境已安装 qdrant-client 时，自动切换为 Qdrant
服务端（生产推荐，支持大规模与持久化）。

对外接口统一：upsert / search / count / reset。
"""
from __future__ import annotations

import httpx
import math
import uuid
from typing import Any

from app.config import Settings, get_settings
from app.log import get_logger

log = get_logger("rag.vectorstore")


def point_id(doc_id: str) -> str:
    """确定性 Qdrant point id（UUID5）。

    不要用 Python 内置 hash()——其字符串哈希带随机 salt，跨进程不一致，
    会让「增量入库」产生重复点、且 point id 不可复现（破坏 upsert 幂等）。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


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

    def delete_by_doc_id(self, doc_id: str):
        """按文档整体下架：删除所有 payload.doc_id 匹配的向量。"""
        self._data = [d for d in self._data if d["payload"].get("doc_id") != doc_id]

    def all_payloads(self) -> list[dict]:
        """返回库内全部 payload，供 BM25 与校验重建使用。"""
        return [d["payload"] for d in self._data]


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

        def delete_by_doc_id(self, doc_id: str):
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            self.client.delete(
                collection_name=self.name,
                points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            )

        def all_payloads(self) -> list[dict]:
            out: list[dict] = []
            next_off = None
            while True:
                pts, next_off = self.client.scroll(
                    collection_name=self.name, limit=256, offset=next_off,
                    with_payload=True, with_vectors=False,
                )
                out.extend(p.payload for p in pts)
                if not next_off:
                    break
            return out

    return QdrantStore()


class QdrantRestStore:
    """无 qdrant-client 依赖的 Qdrant REST 适配（沙箱/轻量环境可用）。

    与 QdrantStore 接口完全一致（upsert/search/count/reset），
    仅通过 httpx 调 Qdrant REST API，免去额外装包。
    """

    def __init__(self, settings: Settings):
        self.s = settings
        self.name = settings.collection_name
        self.base = settings.qdrant_url.rstrip("/")
        self.key = settings.qdrant_api_key or None
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["api-key"] = self.key
        self._http = httpx.Client(timeout=60.0, headers=headers)
        self._ensure()

    def _ensure(self):
        if not self._exists():
            self._create()

    def _exists(self) -> bool:
        r = self._http.get(f"{self.base}/collections/{self.name}")
        return r.status_code == 200

    def _create(self):
        r = self._http.put(
            f"{self.base}/collections/{self.name}",
            json={"vectors": {"size": self.s.vector_dim, "distance": "Cosine"}},
        )
        r.raise_for_status()

    def upsert(self, points: list[dict]):
        if not points:
            return
        batch = [
            {"id": p["id"], "vector": p["vector"], "payload": p["payload"]}
            for p in points
        ]
        step = 200  # 分批，避免单次请求体过大
        for i in range(0, len(batch), step):
            r = self._http.put(
                f"{self.base}/collections/{self.name}/points",
                json={"points": batch[i : i + step]},
            )
            r.raise_for_status()

    def count(self) -> int:
        r = self._http.get(f"{self.base}/collections/{self.name}")
        r.raise_for_status()
        return int(r.json()["result"]["points_count"])

    def search(self, vector: list[float], top_k: int, category: str | None = None):
        body = {"query": vector, "limit": top_k, "with_payload": True}
        if category:
            body["filter"] = {
                "must": [{"key": "category", "match": {"value": category}}]
            }
        r = self._http.post(
            f"{self.base}/collections/{self.name}/points/query", json=body
        )
        r.raise_for_status()
        pts = r.json()["result"]["points"]
        return [
            {"id": p["id"], "score": float(p["score"]), "payload": p.get("payload") or {}}
            for p in pts
        ]

    def reset(self):
        self._http.delete(f"{self.base}/collections/{self.name}")
        self._create()

    def delete_by_doc_id(self, doc_id: str):
        r = self._http.post(
            f"{self.base}/collections/{self.name}/points/delete",
            json={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )
        r.raise_for_status()

    def all_payloads(self) -> list[dict]:
        out: list[dict] = []
        next_off = None
        while True:
            body = {"limit": 256, "with_payload": True, "with_vector": False}
            if next_off is not None:
                body["offset"] = next_off
            r = self._http.post(f"{self.base}/collections/{self.name}/points/scroll", json=body)
            r.raise_for_status()
            res = r.json()["result"]
            out.extend(p.get("payload") or {} for p in res.get("points", []))
            next_off = res.get("next_page_offset")
            if next_off is None:
                break
        return out


def _build_qdrant_rest_store(settings: Settings) -> QdrantRestStore:
    return QdrantRestStore(settings)


class VectorStore:
    """外观类：根据环境与配置选择底层实现。"""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._impl = None
        if self.s.qdrant_url:
            # 优先 qdrant-client；缺失/失败则走零依赖 REST 兜底
            try:
                self._impl = _build_qdrant_store(self.s)
            except Exception as e:  # noqa: BLE001
                log.warning("[向量库] qdrant-client 初始化失败：%s；尝试 REST 兜底", e)
                try:
                    self._impl = _build_qdrant_rest_store(self.s)
                except Exception as e2:  # noqa: BLE001
                    print(f"[向量库] Qdrant REST 也失败，回退纯内存：{e2}")
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

    def delete_by_doc_id(self, doc_id: str):
        self._impl.delete_by_doc_id(doc_id)

    def all_payloads(self) -> list[dict]:
        return self._impl.all_payloads()


_store_instance: "VectorStore | None" = None


def get_vector_store(settings: Settings | None = None) -> "VectorStore":
    """全局单例：保证入库与查询共用同一份内存索引（或同一 Qdrant 连接）。"""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(settings or get_settings())
    return _store_instance
