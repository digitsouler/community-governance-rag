"""意图识别统一层（IntentLayer）。

设计目标：把散落在 pipeline.py / gateway.py 各处的“中文关键词 + 正则”列表
全部抽到 intents.yaml，本文件只负责“匹配逻辑”。这样：

- 维护：加词/改词只改 intents.yaml，不用翻代码；
- 一致性：所有检测器共用一套接口与策略；
- 可测：行为集中，单测一眼覆盖。

行为等价说明：本模块的方法逐一复刻原 _supervise / _infer_role /
_should_decompose / _maybe_recall / _GRATITUDE_PATTERNS / injection /
safety 等逻辑，仅数据源改为配置。算法（如事实抽取的最丰富匹配）仍留在
调用方（pipeline.py），本层只暴露配置与已编译正则。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("intent")

DEFAULT_PATH = Path(__file__).with_name("intents.yaml")

_VALID_ROLES = ("resident", "property", "mediator", "grid_worker")


class IntentLayer:
    def __init__(self, path=DEFAULT_PATH):
        self.cfg = {}
        if path:
            p = Path(path)
            if p.exists():
                self.cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            else:
                log.warning("意图配置文件不存在：%s，将使用空配置（所有检测退化为不命中）", p)
        # 预编译正则
        self._inj_re = [re.compile(p, re.I) for p in self.cfg.get("injection_patterns", [])]
        safety = self.cfg.get("safety", {}) or {}
        self._violent_re = re.compile(safety["violent_regex"], re.I) if safety.get("violent_regex") else None
        self._selfharm_re = re.compile(safety["self_harm_regex"], re.I) if safety.get("self_harm_regex") else None
        self._asked_re = [re.compile(p) for p in self.cfg.get("asked_question_patterns", [])]
        self._fact_time_re = [re.compile(p) for p in self.cfg.get("fact_time_patterns", [])]
        self._fact_dims = self.cfg.get("fact_dimensions", {}) or {}

    # ------------------------------------------------------------------
    # 路由：_supervise 复刻
    # ------------------------------------------------------------------
    def supervise(self, question: str, has_history: bool = False) -> str:
        q = (question or "").strip()
        ql = q.lower()
        if any(g in ql for g in self.cfg.get("greet", [])) and len(q) <= 12:
            return "direct"
        if any(k in q for k in self.cfg.get("bot_intro", [])):
            return "direct"
        if len(q) < 4:
            return "retrieve" if has_history else "clarify"
        gov = self.cfg.get("governance_domain", [])
        off = self.cfg.get("off_domain", [])
        hit_gov = any(k in ql for k in gov)
        hit_off = any(k in ql for k in off)
        if hit_off and not hit_gov:
            return "out_of_domain"
        return "retrieve"

    # ------------------------------------------------------------------
    # 角色推断：_infer_role 复刻（优先级 mediator > property > resident）
    # ------------------------------------------------------------------
    def infer_role(self, question: str, history: list[dict] | None = None) -> str:
        q = question or ""
        if history:
            q = " ".join(m["content"] for m in history if m.get("role") == "user") + " " + q
        role = self.cfg.get("role", {}) or {}
        if any(k in q for k in role.get("mediator", [])):
            return "mediator"
        if any(k in q for k in role.get("property", [])):
            return "property"
        return "resident"

    # ------------------------------------------------------------------
    # 回忆类：_maybe_recall 关键词判定部分复刻（≤60 字短句）
    # ------------------------------------------------------------------
    def detect_recall(self, question: str) -> bool:
        q = (question or "").strip()
        if not q or len(q) > 60:
            return False
        return bool(any(kw in q for kw in self.cfg.get("recall", [])))

    # ------------------------------------------------------------------
    # 查询分解：_should_decompose 复刻
    # ------------------------------------------------------------------
    def should_decompose(self, question: str) -> bool:
        d = self.cfg.get("decompose", {}) or {}
        q = (question or "").strip()
        log.info("[decompose-debug] q_len=%d qmark=%d qwords=%d | q=%r",
                 len(q), q.count("?") + q.count("？"),
                 sum(q.count(w) for w in d.get("question_words", [])), q[:50])
        if len(q) < d.get("min_len", 36):
            return False
        if q.count("?") + q.count("？") >= 2:
            return True
        qwords = sum(q.count(w) for w in d.get("question_words", []))
        if qwords >= d.get("min_qwords", 2):
            return True
        if any(c in q for c in d.get("conjunctions", [])) and any(k in q for k in d.get("triggers", [])):
            return True
        return False

    # ------------------------------------------------------------------
    # 感谢短路：_generate_stream 内 _GRATITUDE_PATTERNS 复刻（<30 字）
    # ------------------------------------------------------------------
    def detect_gratitude(self, question: str) -> bool:
        q = question or ""
        if len(q) >= 30:
            return False
        return any(p in q for p in self.cfg.get("gratitude", []))

    # ------------------------------------------------------------------
    # 危险意图：gateway.HARM_ACTION_RE / SELF_HARM_RE 复刻
    # 返回 None / "self_harm" / "violent_intent"
    # ------------------------------------------------------------------
    def detect_safety(self, text: str) -> Optional[str]:
        t = text or ""
        if self._selfharm_re and self._selfharm_re.search(t):
            return "self_harm"
        if self._violent_re and self._violent_re.search(t):
            return "violent_intent"
        return None

    def match_injection(self, text: str) -> bool:
        return any(rx.search(text or "") for rx in self._inj_re)

    def safety_response(self, reason: Optional[str]) -> Optional[str]:
        return (self.cfg.get("safety_responses", {}) or {}).get(reason or "")

    # ------------------------------------------------------------------
    # 检索前清洗：_reformulate 复刻（去停用词）
    # ------------------------------------------------------------------
    def reformulate(self, query: str) -> str:
        q = query or ""
        for w in self.cfg.get("stopwords", []):
            q = q.replace(w, "")
        return q.strip() or query

    # ------------------------------------------------------------------
    # 暴露给 pipeline 的复杂抽取所需数据
    # ------------------------------------------------------------------
    @property
    def fact_dimensions(self) -> dict:
        return self._fact_dims

    @property
    def asked_regexes(self) -> list[re.Pattern]:
        return self._asked_re

    @property
    def fact_time_regexes(self) -> list[re.Pattern]:
        return self._fact_time_re

    def fact_time_score(self, s: str) -> int:
        """事实抽取“起止时间”维度选最丰富候选的评分（原 _rich_score 复刻）。"""
        score = len(s)
        if re.search(r"\d+[点:时：]", s):
            score += 50
        if any(w in s for w in self.cfg.get("fact_time_night_markers", [])):
            score += 40
        if any(w in s for w in self.cfg.get("fact_time_range_markers", [])):
            score += 20
        return score


_layer: Optional[IntentLayer] = None


def get_intent_layer() -> IntentLayer:
    global _layer
    if _layer is None:
        _layer = IntentLayer()
    return _layer
