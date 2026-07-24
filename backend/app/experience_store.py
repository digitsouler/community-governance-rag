"""长期记忆层（P3）：跨会话案件经验向量库。

把每个会话沉淀出的「案件画像」向量化存入 Qdrant（独立集合 mediation_experiences），
新对话时按语义检索最相似的若干历史案件，作为「相关历史经验」注入当前回答——
实现跨会话的经验复用：处理相似矛盾时，建议口径与处置思路保持一致、可借鉴。

存储分层：
  - 结构化画像/案件档案 → Redis（case:profile:{sid}），见 profile_store.py
  - 需要「凭意思找相似」的模糊经验 → Qdrant（语义检索），本模块

降级：无 qdrant_url 或连接失败 → 内存模式（进程内 dict），保证任意环境可跑。
"""
from __future__ import annotations

import httpx
import uuid

from app.config import get_settings
from app.log import get_logger

log = get_logger("experience_store")

COLLECTION = "mediation_experiences"


def _point_id(sid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "exp:" + sid))


class ExperienceStore:
    def __init__(self, settings=None) -> None:
        self.s = settings or get_settings()
        self.name = COLLECTION
        self._mem: dict[str, dict] = {}
        self.mode = "memory"
        self._http = None
        url = self.s.qdrant_url
        if url:
            try:
                self.base = url.rstrip("/")
                self.key = self.s.qdrant_api_key or None
                headers = {"Content-Type": "application/json"}
                if self.key:
                    headers["api-key"] = self.key
                self._http = httpx.Client(timeout=10.0, headers=headers)
                self._create_if_missing()
                self.mode = "qdrant"
                log.info("长期经验库：Qdrant 模式 | %s/%s", url, self.name)
            except Exception as e:  # noqa: BLE001
                self.mode = "memory"
                self._http = None
                log.warning("长期经验库：Qdrant 不可用，退回内存模式（%s）", e)

    def _create_if_missing(self):
        assert self._http is not None
        if self._http.get(f"{self.base}/collections/{self.name}").status_code == 200:
            return
        self._http.put(
            f"{self.base}/collections/{self.name}",
            json={"vectors": {"size": self.s.vector_dim, "distance": "Cosine"}},
        ).raise_for_status()

    def upsert(self, sid: str, vector: list[float], payload: dict) -> None:
        pid = _point_id(sid)
        if self.mode == "qdrant":
            assert self._http is not None
            self._http.put(
                f"{self.base}/collections/{self.name}/points",
                json={"points": [{"id": pid, "vector": vector, "payload": payload}]},
            ).raise_for_status()
        else:
            self._mem[pid] = {"vector": vector, "payload": payload}

    def search(self, vector: list[float], top_k: int = 3, exclude_sid: str = "") -> list[dict]:
        if self.mode == "qdrant":
            return self._search_qdrant(vector, top_k, exclude_sid)
        return self._search_mem(vector, top_k, exclude_sid)

    def _search_qdrant(self, vector, top_k, exclude_sid) -> list[dict]:
        assert self._http is not None
        body = {"query": vector, "limit": top_k, "with_payload": True}
        if exclude_sid:
            body["filter"] = {
                "must_not": [{"key": "session_id", "match": {"value": exclude_sid}}]
            }
        r = self._http.post(
            f"{self.base}/collections/{self.name}/points/query", json=body
        )
        r.raise_for_status()
        return [p.get("payload") or {} for p in r.json()["result"]["points"]]

    def _search_mem(self, vector, top_k, exclude_sid) -> list[dict]:
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = [
            (cos(vector, v["vector"]), v["payload"])
            for v in self._mem.values()
            if v["payload"].get("session_id") != exclude_sid
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def delete(self, sid: str) -> None:
        pid = _point_id(sid)
        if self.mode == "qdrant":
            assert self._http is not None
            self._http.post(
                f"{self.base}/collections/{self.name}/points/delete",
                json={"points": [pid]},
            )
        else:
            self._mem.pop(pid, None)


_store: ExperienceStore | None = None


def get_experience_store() -> ExperienceStore:
    global _store
    if _store is None:
        _store = ExperienceStore()
    return _store
