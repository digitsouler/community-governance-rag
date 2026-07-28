"""用户级长期记忆层（B 方案）：
1. question_log：跨 session、跨角色汇总的「用户最近提问」索引，供「我刚才问了什么」类问题回答用
2. 案件档案改为 (user_id, role) 双键隔离：同一角色跨 session 共享专业记忆，跨角色不串

设计原则：复用 session_store / profile_store 的 Redis-or-内存降级模式，零新依赖。
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any

from app.log import get_logger

log = get_logger("user_memory")


# 单条 question_log 记录最大保留条数（按用户；按时间倒序裁剪）
QUESTION_LOG_LIMIT = 50


class UserMemory:
    """用户级长期记忆：question_log + role-scoped profile 索引。

    数据结构（内存模式）：
      _questions: dict[user_id] -> deque[{ts, session_id, role, question, preview}]  按时间正序，左 pop 旧
      _profiles:  dict[(user_id, role)] -> dict（案件档案，替代原 session_id key）
    Redis 模式：
      user:{uid}:questions        List<json>（json 字符串）
      case:profile:{uid}:{role}   String<json>
    """

    def __init__(self) -> None:
        self._mode = "memory"
        self._redis = None
        try:
            import redis  # 延迟导入
            from app.config import get_settings

            url = get_settings().redis_url or ""
            if url:
                self._redis = redis.from_url(
                    url, socket_connect_timeout=1.0, socket_timeout=1.0,
                    decode_responses=True, protocol=2,
                )
                self._redis.ping()
                self._mode = "redis"
                log.info("用户记忆：Redis 模式 | %s", url)
        except Exception as e:  # noqa: BLE001
            self._mode = "memory"
            self._redis = None
            log.warning("用户记忆：Redis 不可用，退回内存模式（%s）", e)
        if self._mode == "memory":
            self._questions: dict[str, deque] = {}
            self._profiles: dict[tuple[str, str], dict] = {}
            self._lock = threading.Lock()

    # ---------- 公共属性 ----------
    @property
    def mode(self) -> str:
        return self._mode

    # ---------- question_log ----------
    def _q_key(self, uid: str) -> str:
        return f"user:{uid}:questions"

    def record_question(self, user_id: str, session_id: str, role: str, question: str) -> None:
        """记录一条用户提问到用户级 question_log（跨 session/角色共享）。"""
        if not user_id or not question:
            return
        item = {
            "ts": time.time(),
            "session_id": session_id or "",
            "role": role or "",
            "question": question,
            "preview": question.strip().replace("\n", " ")[:80],
        }
        if self._mode == "redis":
            assert self._redis is not None
            self._redis.rpush(self._q_key(user_id), json.dumps(item, ensure_ascii=False))
            # 限长：超出保留窗口的旧条目裁剪（按 LRU 头部裁剪）
            self._redis.ltrim(self._q_key(user_id), -QUESTION_LOG_LIMIT, -1)
            return
        with self._lock:
            dq = self._questions.setdefault(user_id, deque(maxlen=QUESTION_LOG_LIMIT))
            dq.append(item)

    def get_recent_questions(self, user_id: str, role: str | None = None, n: int = 10) -> list[dict]:
        """取最近 n 条用户提问（按时间倒序）。

        role 非空时仅返回该角色的记录（用于"我刚才以 X 身份问过什么"）。
        """
        if not user_id:
            return []
        n = max(1, min(int(n or 10), QUESTION_LOG_LIMIT))
        if self._mode == "redis":
            assert self._redis is not None
            raws = self._redis.lrange(self._q_key(user_id), 0, -1)
            items = []
            for r in raws:
                try:
                    items.append(json.loads(r))
                except Exception:
                    continue
        else:
            with self._lock:
                items = list(self._questions.get(user_id, []))
        if role:
            items = [it for it in items if (it.get("role") or "") == role]
        items.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return items[:n]

    # ---------- 案件档案（按 user_id + role 隔离） ----------
    def _p_key(self, uid: str, role: str) -> str:
        return f"case:profile:{uid}:{role or 'unknown'}"

    def get_profile(self, user_id: str, role: str) -> dict:
        if not user_id:
            return {}
        key = self._p_key(user_id, role or "unknown")
        if self._mode == "redis":
            assert self._redis is not None
            raw = self._redis.get(key)
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except Exception:
                return {}
        with self._lock:
            return dict(self._profiles.get((user_id, role or "unknown"), {}))

    def save_profile(self, user_id: str, role: str, profile: dict) -> None:
        if not user_id:
            return
        profile = dict(profile)
        profile["updated_at"] = time.time()
        key = self._p_key(user_id, role or "unknown")
        if self._mode == "redis":
            assert self._redis is not None
            self._redis.set(key, json.dumps(profile, ensure_ascii=False))
            return
        with self._lock:
            self._profiles[(user_id, role or "unknown")] = profile


_store: UserMemory | None = None


def get_user_memory() -> UserMemory:
    global _store
    if _store is None:
        _store = UserMemory()
    return _store
