"""短期记忆层：会话存储。

设计目标（贴合本项目「零依赖、可降级」原则）：
- 若环境装了 redis 且配置了 REDIS_URL，则使用 Redis 持久化会话（跨重启/跨标签共享）；
- 否则自动退回内存模式（进程内 dict），保证无 Redis 时后端照常启动、可用于本地开发/沙箱验证。

Redis 数据结构：
  session:{id}          → List<str(json msg)>        当前会话的完整消息流
  session:meta:{id}     → Hash {title,created_at,updated_at,msg_count}
  sessions              → ZSet(score=updated_at, member=id)  用于列表按时间倒序

消息格式：
  {"role": "user"|"assistant", "content": str, "ts": float}
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from app.log import get_logger

log = get_logger("session_store")


class SessionStore:
    def __init__(self) -> None:
        self._mode = "memory"
        self._redis = None
        try:
            import redis  # 延迟导入：无 redis 客户端的沙箱可降级

            url = self._resolve_url()
            if url:
                self._redis = redis.from_url(
                    url, socket_connect_timeout=1.0, socket_timeout=1.0,
                    decode_responses=True, protocol=2,
                )
                self._redis.ping()
                self._mode = "redis"
                log.info("会话存储：Redis 模式 | %s", url)
        except Exception as e:  # noqa: BLE001
            self._mode = "memory"
            self._redis = None
            log.warning("会话存储：Redis 不可用，退回内存模式（%s）", e)

        if self._mode == "memory":
            self._mem: dict[str, list[dict]] = {}
            self._meta: dict[str, dict] = {}
            self._lock = threading.Lock()

    # ---------- 配置 ----------
    @staticmethod
    def _resolve_url() -> str:
        try:
            from app.config import get_settings

            return get_settings().redis_url or ""
        except Exception:  # noqa: BLE001
            return ""

    @property
    def mode(self) -> str:
        return self._mode

    # ---------- 写 ----------
    def create_session(self, title: str = "", owner: str = "") -> str:
        sid = uuid.uuid4().hex[:16]
        now = time.time()
        meta = {"title": title, "created_at": now, "updated_at": now, "msg_count": 0, "owner": owner}
        if self._mode == "redis":
            assert self._redis is not None
            self._redis.delete(f"session:{sid}")
            key = f"session:meta:{sid}"
            for k, v in meta.items():
                self._redis.hset(key, k, str(v))
            self._redis.zadd("sessions", {sid: now})
        else:
            with self._lock:
                self._mem[sid] = []
                self._meta[sid] = meta
        return sid

    def append_message(self, sid: str, role: str, content: str) -> None:
        msg = {"role": role, "content": content, "ts": time.time()}
        if self._mode == "redis":
            assert self._redis is not None
            # 首次用户消息自动作为标题（取前 24 字）
            meta = self._redis.hgetall(f"session:meta:{sid}")
            if role == "user" and (not meta.get("title")):
                title = content.strip().replace("\n", " ")[:24]
                self._redis.hset(f"session:meta:{sid}", "title", title)
            self._redis.rpush(f"session:{sid}", json.dumps(msg, ensure_ascii=False))
            self._redis.hincrby(f"session:meta:{sid}", "msg_count", 1)
            self._redis.hset(f"session:meta:{sid}", "updated_at", str(time.time()))
            self._redis.zadd("sessions", {sid: time.time()})
        else:
            with self._lock:
                self._mem.setdefault(sid, [])
                self._meta.setdefault(
                    sid, {"title": "", "created_at": time.time(), "updated_at": time.time(), "msg_count": 0}
                )
                if role == "user" and not self._meta[sid]["title"]:
                    self._meta[sid]["title"] = content.strip().replace("\n", " ")[:24]
                self._mem[sid].append(msg)
                self._meta[sid]["msg_count"] += 1
                self._meta[sid]["updated_at"] = time.time()

    # ---------- 读 ----------
    def get_messages(self, sid: str) -> list[dict]:
        """返回该会话的全部消息（按时间正序）。"""
        if self._mode == "redis":
            assert self._redis is not None
            raws = self._redis.lrange(f"session:{sid}", 0, -1)
            return [json.loads(r) for r in raws]
        with self._lock:
            return list(self._mem.get(sid, []))

    def get_session(self, sid: str) -> dict[str, Any] | None:
        if self._mode == "redis":
            assert self._redis is not None
            meta = self._redis.hgetall(f"session:meta:{sid}")
            if not meta:
                return None
            return {
                "id": sid,
                "title": meta.get("title", ""),
                "owner": meta.get("owner", ""),
                "created_at": float(meta.get("created_at", 0)),
                "updated_at": float(meta.get("updated_at", 0)),
                "msg_count": int(meta.get("msg_count", 0)),
            }
        with self._lock:
            meta = self._meta.get(sid)
            if not meta:
                return None
            return {"id": sid, **meta}

    def list_sessions(self, owner: str = "") -> list[dict]:
        """按 updated_at 倒序返回会话概览；传 owner 时仅返回该用户的会话（多用户隔离）。"""
        if self._mode == "redis":
            assert self._redis is not None
            ids = self._redis.zrevrange("sessions", 0, -1)
            out = []
            for sid in ids:
                m = self.get_session(sid)
                if m and (not owner or m.get("owner") == owner):
                    out.append(m)
            return out
        with self._lock:
            items = ({"id": sid, **meta} for sid, meta in self._meta.items())
            if owner:
                items = [m for m in items if m.get("owner") == owner]
            return sorted(items, key=lambda x: x["updated_at"], reverse=True)

    # ---------- 删除 ----------
    def delete_session(self, sid: str) -> bool:
        if self._mode == "redis":
            assert self._redis is not None
            if not self._redis.exists(f"session:meta:{sid}"):
                return False
            self._redis.delete(f"session:{sid}")
            self._redis.delete(f"session:meta:{sid}")
            self._redis.zrem("sessions", sid)
            return True
        with self._lock:
            if sid not in self._meta:
                return False
            self._mem.pop(sid, None)
            self._meta.pop(sid, None)
            return True


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
