"""网关层：多用户隔离 / 限流 / 护栏 / 审计 / 语义缓存。

设计原则（贴合本项目「零依赖、可降级」）：
- 仅依赖已安装的 httpx；Redis 可选，不可用时自动退回内存模式。
- 所有能力均可在 config 中开关，默认开启。
- 作为 main.py 与各能力之间的薄中间件，不改动 RAG 管道核心逻辑。

请求流（main.py 中按序调用）：
  接入 → 解析 user_id → 限流(IP+用户+全局) → 输入护栏 → 语义缓存 → 管道 → 输出护栏 → 审计
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Optional

from app.config import get_settings
from app.log import get_logger

log = get_logger("gateway")
_s = get_settings()


def error_payload(msg: str, trace_id: str = "", code: int = 400) -> dict:
    """统一错误结构：所有网关拒绝/异常都返回 {error, trace_id, code}。"""
    return {"error": msg, "trace_id": trace_id, "code": code}


# ---------- 用户解析（多用户隔离入口） ----------
def resolve_user(headers: dict, client_ip: str = "") -> str:
    """从 X-User-Id 或 Authorization(bearer) 解析用户身份；缺省则按 IP 匿名。"""
    uid = (headers.get("X-User-Id") or "").strip()
    if uid:
        return uid[:64]
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()[:64]
    return f"anon:{client_ip}"


# ---------- Redis 句柄（懒加载，与 session_store 同款连接参数） ----------
_rds: Any = None
_rds_lock = threading.Lock()


def get_redis() -> Any:
    global _rds
    if _rds is not None:
        return _rds
    with _rds_lock:
        if _rds is not None:
            return _rds
        try:
            import redis

            if _s.redis_url:
                r = redis.from_url(
                    _s.redis_url, socket_connect_timeout=1.0,
                    socket_timeout=1.0, decode_responses=True, protocol=2,
                )
                r.ping()
                _rds = r
                log.info("网关：Redis 模式（限流/审计可持久化）")
            else:
                log.info("网关：Redis 未配置，内存兜底")
        except Exception as e:  # noqa: BLE001
            log.warning("网关：Redis 不可用，退回内存模式（%s）", e)
            _rds = None
    return _rds


# ---------- 限流（固定窗口计数器，Redis 或内存） ----------
class RateLimiter:
    def __init__(self) -> None:
        self.r = get_redis()
        self.mode = "redis" if self.r else "memory"
        self._mem: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """返回 (allowed, retry_after_seconds)。fail-open：异常时放行。"""
        if not _s.rate_limit_enabled:
            return True, 0
        if self.mode == "redis":
            assert self.r is not None
            full = f"rl:{key}"
            try:
                n = self.r.incr(full)
                if n == 1:
                    self.r.expire(full, window)
                if n > limit:
                    return False, max(1, self.r.ttl(full) or window)
                return True, 0
            except Exception as e:  # noqa: BLE001
                log.warning("限流 Redis 异常，放行：%s", e)
                return True, 0
        # 内存固定窗口
        now = time.time()
        with self._lock:
            hits = self._mem.setdefault(key, [])
            hits = [t for t in hits if now - t < window]
            if len(hits) >= limit:
                retry = int(window - (now - hits[0])) + 1
                self._mem[key] = hits
                return False, max(1, retry)
            hits.append(now)
            self._mem[key] = hits
            return True, 0


_rl: Optional[RateLimiter] = None


def get_ratelimiter() -> RateLimiter:
    global _rl
    if _rl is None:
        _rl = RateLimiter()
    return _rl


# ---------- 护栏：prompt 注入检测 + PII 脱敏 ----------
INJECTION_PATTERNS = [
    r"忽略(以上|前面|之前|上述|之前所有).{0,8}(指令|提示|规定|规则|系统|要求|设定)",
    r"ignore\s+(the\s+)?(previous|above|prior|system|all)\s+(instructions|prompt|rules)",
    r"disregard\s+.{0,30}instructions",
    r"(你|你现在|现在).{0,6}(是|变成|切换|进入).{0,10}(开发|调试|开发者|管理员|root|system|后台|danzel?)模式",
    r"(开发者|开发|调试|越狱|jailbreak|developer)\s*模式",
    r"(repeat|output|print|输出|复述|打印|透露)\s*(your|你(的)?)\s*(system\s+)?(prompt|提示词|系统提示|设定)",
    r"system\s+prompt",
    r"把(你|系统)的(系统)?提示",
    r"作为(一个)?(开发|调试|测试)人员",
]
_INJ_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

# 高风险意图：只有「实施/准备实施的动作倾向」与「严重伤害手段或目标」同时出现才拦截。
# 单纯讨论新闻、法律责任、调解案例中的“枪炮战火”等词，不应仅凭关键词误伤。
HARM_ACTION_RE = re.compile(
    r"(我想|我要|我准备|我打算|计划|决定|马上|现在就|帮我|教我|告诉我怎么|如何|怎么).{0,18}"
    r"(杀|砍|捅|打死|弄死|炸|放火|纵火|枪击|袭击|报复|投毒|绑架|制造爆炸|发动战争|实施犯罪|搞大事)"
    r"|"
    r"(杀|砍|捅|打死|弄死|炸|放火|纵火|枪击|袭击|报复|投毒|绑架|制造爆炸|发动战争|实施犯罪|搞大事).{0,18}"
    r"(怎么办|怎么做|步骤|方法|计划|方案|工具|材料)",
    re.I,
)
SELF_HARM_RE = re.compile(
    r"(不想活|活不下去|结束生命|自杀|轻生|跳楼|割腕|服毒|寻死|去死).{0,20}"
    r"(我|自己|本人|现在|马上|准备|打算|想|要|方法|怎么)",
    re.I,
)

HIGH_RISK_RESPONSES = {
    "violent_intent": (
        "我不能帮助策划、实施或美化伤害他人、枪击、纵火、爆炸、恐怖袭击或其他犯罪行为。"
        "请立即停止当前行动，远离武器、易燃易爆物和可能受伤的人，不要单独处理。"
        "如果存在现实紧迫危险，请马上拨打 110；如有人受伤，请同时拨打 120。"
        "可以先联系一位可信任的家人、同事或社区负责人陪同，并只描述冲突事实，我可以继续帮你整理一份合法、安全的降温和求助步骤。"
    ),
    "self_harm": (
        "我很重视你现在的安全，但不能帮助提供自伤或结束生命的方法。"
        "请先远离可能伤害自己的物品和高处，尽快联系一位可信任的人陪在身边。"
        "如果你正准备行动或无法保证自己安全，请立即拨打 110 或 120，或直接前往最近的急诊。"
        "你也可以只告诉我：你现在是否安全、身边是否有人，我会继续陪你把下一步求助安排清楚。"
    ),
}

PII_PHONE = re.compile(r"(?<![\d.])(1[3-9]\d)\d{4}(\d{4})(?![\d.])")
PII_IDCARD = re.compile(r"(?<![\d.])\d{6}(\d{8})\d{4}(?![\d.])")
PII_EMAIL = re.compile(r"([a-zA-Z0-9_.]{1,3})[a-zA-Z0-9_.]*@([a-zA-Z0-9_.-]+\.[a-zA-Z]{2,})")


class Guardrails:
    def __init__(self) -> None:
        self.enabled = _s.guardrails_enabled

    def scan_input(self, text: str) -> tuple[bool, Optional[str], bool]:
        """返回 (ok, block_reason, has_pii)。输入 PII 不阻断（当事人需描述案情），仅标记。"""
        if not self.enabled:
            return True, None, False
        has_pii = bool(PII_PHONE.search(text) or PII_IDCARD.search(text) or PII_EMAIL.search(text))
        if SELF_HARM_RE.search(text):
            return False, "self_harm", has_pii
        if HARM_ACTION_RE.search(text):
            return False, "violent_intent", has_pii
        for rx in _INJ_RE:
            if rx.search(text):
                return False, "prompt_injection", has_pii
        return True, None, has_pii

    @staticmethod
    def safety_response(reason: Optional[str]) -> Optional[str]:
        """高风险意图返回确定性劝阻文案；prompt 注入等普通阻断仍返回 None。"""
        return HIGH_RISK_RESPONSES.get(reason or "")

    def redact(self, text: str) -> str:
        if not (self.enabled and _s.pii_redact_enabled):
            return text
        t = PII_PHONE.sub(r"\1****\2", text)
        t = PII_IDCARD.sub(lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:], t)
        t = PII_EMAIL.sub(lambda m: m.group(1) + "***@" + m.group(2), t)
        return t


_gr: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    global _gr
    if _gr is None:
        _gr = Guardrails()
    return _gr


# ---------- 语义缓存（查询向量近邻命中 → 短路返回） ----------
class SemanticCache:
    def __init__(self) -> None:
        self.enabled = _s.semantic_cache_enabled
        self.threshold = _s.semantic_cache_sim
        self.max = _s.semantic_cache_size
        self._mem: list[tuple[list[float], str]] = []  # (vec, answer)
        self._lock = threading.Lock()
        self._emb = None

    def _embed(self, text: str) -> list[float]:
        if self._emb is None:
            from app.rag.embeddings import EmbeddingClient

            self._emb = EmbeddingClient()
        return self._emb.embed_query(text)

    def get(self, question: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            vec = self._embed(question)
        except Exception as e:  # noqa: BLE001
            log.warning("语义缓存嵌入失败，跳过：%s", e)
            return None
        with self._lock:
            best: Optional[str] = None
            best_sim = -1.0
            for v, ans in self._mem:
                sim = self._cosine(vec, v)
                if sim > best_sim:
                    best_sim, best = sim, ans
            if best is not None and best_sim >= self.threshold:
                log.info("语义缓存命中 | sim=%.3f", best_sim)
                return best
        return None

    def put(self, question: str, answer: str, has_pii: bool) -> None:
        if not self.enabled or has_pii:
            return  # 含 PII 的答案不缓存，避免泄露
        try:
            vec = self._embed(question)
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._mem.append((vec, answer))
            if len(self._mem) > self.max:
                self._mem.pop(0)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


_sc: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    global _sc
    if _sc is None:
        _sc = SemanticCache()
    return _sc


# ---------- 审计（append-only，走结构化日志） ----------
def audit(event: str, **fields: Any) -> None:
    if not _s.audit_enabled:
        return
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    log.info("AUDIT %s", json.dumps(rec, ensure_ascii=False))
