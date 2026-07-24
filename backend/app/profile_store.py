"""长期记忆层（P1）：结构化案件档案。

把一段对话里提取出的「案件类型 / 对方当事人 / 已知事实 / 证据情况 / 当前阶段」
以结构化 JSON 存进 Redis（或内存降级），作为 AI 回答的「事实源」。
核心价值：即使原始多轮对话被上下文窗口截断，结构化档案仍在，AI 据此作答而非瞎编。

Redis 键：
  case:profile:{session_id}  →  JSON string（完整档案）
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from app.log import get_logger

log = get_logger("profile_store")


class ProfileStore:
    def __init__(self) -> None:
        self._mode = "memory"
        self._redis = None
        try:
            import redis  # 延迟导入，无客户端则降级

            from app.config import get_settings

            url = get_settings().redis_url or ""
            if url:
                self._redis = redis.from_url(
                    url, socket_connect_timeout=1.0, socket_timeout=1.0,
                    decode_responses=True, protocol=2,
                )
                self._redis.ping()
                self._mode = "redis"
                log.info("案件档案：Redis 模式 | %s", url)
        except Exception as e:  # noqa: BLE001
            self._mode = "memory"
            self._redis = None
            log.warning("案件档案：Redis 不可用，退回内存模式（%s）", e)
        if self._mode == "memory":
            self._mem: dict[str, dict] = {}
            self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    def _key(self, sid: str) -> str:
        return f"case:profile:{sid}"

    def get(self, sid: str) -> dict:
        if not sid:
            return {}
        if self._mode == "redis":
            assert self._redis is not None
            raw = self._redis.get(self._key(sid))
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except Exception:
                return {}
        with self._lock:
            return dict(self._mem.get(sid, {}))

    def save(self, sid: str, profile: dict) -> None:
        if not sid:
            return
        profile = dict(profile)
        profile["updated_at"] = time.time()
        if self._mode == "redis":
            assert self._redis is not None
            self._redis.set(self._key(sid), json.dumps(profile, ensure_ascii=False))
        else:
            with self._lock:
                self._mem[sid] = profile

    def delete(self, sid: str) -> None:
        if not sid:
            return
        if self._mode == "redis":
            assert self._redis is not None
            self._redis.delete(self._key(sid))
        else:
            with self._lock:
                self._mem.pop(sid, None)


_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store
