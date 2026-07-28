"""核心 RAG 管道（Agentic）。

两大差异化能力：
  1. Supervisor 路由：先判断问题类型，决定「直接回答 / 要求澄清 / 走检索」，
     避免所有问题都无脑检索（省 token、降幻觉）。
  2. Self-RAG 自纠错：检索结果不达标时自动改写查询重试；重试后仍不足则
     诚实告知「知识库暂无依据」，绝不编造。

对外暴露 query()：输入问题 + 模型供应商，输出答案、引用来源、路由决策与重试次数。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from app.config import ProviderName, Settings, get_settings
from app.log import get_logger
from app.profile_store import get_profile_store
from app.user_memory import get_user_memory
from app.rag.embeddings import EmbeddingClient
from app.rag.hybrid import BM25Index, rrf_fuse
from app.rag.llm import LLMClient
from app.rag.rerank import rerank
from app.rag.vectorstore import get_vector_store

log = get_logger("rag.pipeline")

STOPWORDS = ["我想问", "请问", "帮我", "怎么", "如何", "怎么办", "吗", "呢", "？", "?", "。", "居民", "社区", "小区"]

# 社区治理 / 矛盾调解领域的核心关键词（命中则优先判定为域内问题）
GOVERNANCE_KEYWORDS = [
    "邻居", "邻里", "漏水", "噪音", "噪声", "停车", "车位", "地锁", "宠物", "狗", "猫",
    "物业", "物业费", "业委会", "业主大会", "维修基金", "绿地", "违建", "搭建",
    "油烟", "装修", "扰民", "垃圾", "环境", "路灯", "充电桩", "电梯", "群租",
    "出租", "房东", "租客", "租户", "赡养", "抚养", "家暴", "家庭暴力", "纠纷",
    "调解", "矛盾", "投诉", "维权", "居委会", "村委会", "网格员", "社区", "小区",
    "业主", "住户", "公共区域", "共有部分", "采光", "通风", "排水", "排污",
]

# 明显离域的生活 / 娱乐 / 工具类诉求（命中且无治理关键词 → 直接判定为超出范围）
OFF_DOMAIN_KEYWORDS = [
    "ktv", "k歌", "唱歌", "歌厅", "酒吧", "电影", "追剧", "电视剧", "综艺",
    "旅游", "景点", "景区", "爬山", "美食", "餐厅", "饭店", "外卖", "奶茶",
    "快递", "打车", "滴滴", "出租车", "导航", "地图", "天气", "股票", "基金",
    "彩票", "炒币", "游戏", "王者", "原神", "购物", "淘宝", "京东", "拼多多",
    "演唱会", "酒店", "机票", "火车票", "高铁票", "笑话", "算命", "运势", "星座",
    "八卦", "新闻", "翻译", "写代码", "编程",
]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.embedder = EmbeddingClient(self.s)
        self.store = get_vector_store(self.s)
        self.llm = LLMClient(self.s)
        # 稀疏召回索引（BM25）：与向量库同源语料。
        # 启动时若向量库已有数据（来自上一次运行持久化）则直接据此构建；
        # 知识库就绪后由 _ensure_kb() 调 rebuild_bm25() 重建为最新发布状态。
        self.bm25 = BM25Index()
        try:
            payloads = self.store.all_payloads()
            if payloads:
                self.bm25.build([{"id": p["id"], "payload": p} for p in payloads])
                log.info("BM25 稀疏索引构建完成 | 文档数=%d", self.bm25._n)
            else:
                log.info("向量库为空，BM25 待知识库就绪后构建")
        except Exception as e:
            log.warning("BM25 索引构建失败，混合检索降级为纯向量：%s", e)
            self.bm25._built = False
        # 长期记忆（P1）：结构化案件档案存储（Redis 优先，内存降级）
        self._profiles = get_profile_store()
        # 用户级长期记忆（B 方案）：跨 session 跨角色的 question_log + 角色维度的专业档案
        self._user_mem = get_user_memory()

    def rebuild_bm25(self):
        """用向量库当前全部 payload 重建稀疏索引，使其与检索源完全一致。

        在知识库发布/下架后调用，保证 BM25 与稠密向量命中一致的候选池。
        """
        try:
            store = get_vector_store(self.s)
            payloads = store.all_payloads()
            self.bm25.build([{"id": p["id"], "payload": p} for p in payloads])
            log.info("BM25 稀疏索引重建完成 | 文档数=%d", self.bm25._n)
        except Exception as e:  # noqa: BLE001
            log.warning("BM25 重建失败：%s", e)
            self.bm25._built = False
    def _supervise(self, question: str, has_history: bool = False) -> str:
        q = question.strip()
        ql = q.lower()
        greet = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "感谢"]
        if any(g in ql for g in greet) and len(q) <= 12:
            return "direct"
        if any(k in q for k in ["你是谁", "你是什么", "你能干", "你会", "介绍下你", "怎么用"]):
            return "direct"
        # 有历史时，短句多为对上一轮追问的回答（如"好几天了""是的"），继续走检索而非再次澄清
        if len(q) < 4:
            return "retrieve" if has_history else "clarify"
        # 领域判断：命中治理关键词 → 域内（走检索）；
        # 仅命中离域关键词、且无治理关键词 → 超出服务范围（不检索）
        hit_governance = any(k in ql for k in GOVERNANCE_KEYWORDS)
        hit_off_domain = any(k in ql for k in OFF_DOMAIN_KEYWORDS)
        if hit_off_domain and not hit_governance:
            return "out_of_domain"
        return "retrieve"

    def _direct_answer(self) -> str:
        return (
            "我是社区矛盾调解助理，专注于邻里纠纷、物业矛盾、家庭赡养等"
            "社区治理场景的调解支持。你可以直接描述遇到的矛盾（例如"
            "「楼上漏水导致我家天花板发霉怎么办」），我会结合知识库给出"
            "处置建议、相关法条与调解步骤，并标注依据来源。"
        )

    def _out_of_domain(self) -> str:
        return (
            "您的问题超出了我的服务范围。我是社区矛盾调解助理，"
            "专注于邻里纠纷、物业矛盾、家庭赡养、公共设施使用等"
            "社区治理场景的调解支持。如果你遇到的是社区或邻里相关的问题，"
            "请告诉我具体情况，我来帮你检索处置依据。"
        )

    def _clarify(self) -> str:
        return (
            "您的问题信息较少，难以精准检索。请补充：① 矛盾类型"
            "（噪音/漏水/停车/宠物/物业费/赡养等）；② 关键事实"
            "（谁、什么行为、造成什么影响）。例如：「一楼私装地锁占用公共车位，"
            "其他业主该如何处理？」"
        )

    # ---------- 多轮对话辅助 ----------
    def _normalize_history(self, history: list[dict] | None) -> list[dict]:
        """规整前端传来的历史：只保留 user/assistant 的非空文本，取最近 MAX_HISTORY 条。

        兼容前端的 role='bot'（统一转 'assistant'）；过滤 loading/空串；
        末尾若恰好等于本轮问题（前端可能已 push）则不在此处理，由调用方保证不重复。
        """
        MAX_HISTORY = 8  # 最近 8 条 ≈ 4 轮，足够承接语境又不撑爆上下文
        if not history or not isinstance(history, list):
            return []
        norm = []
        for m in history:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content or role not in ("user", "bot", "assistant"):
                continue
            norm.append({"role": "user" if role == "user" else "assistant", "content": content})
        return norm[-MAX_HISTORY:]

    def _contextual_query(self, question: str, history: list[dict]) -> str:
        """把历史里最近的用户话题词并进当前问题，形成用于检索的合并查询。

        追答常是碎片（"好几天了 找过他没用"），本身缺少主题词（"噪音"），
        直接检索会跑题。这里取最近最多 2 条历史【用户】发言拼在当前问题前，
        让稠密/稀疏检索都能锚定到原始话题。仅用于检索，不改变展示给用户的问题。
        """
        if not history:
            return question
        prev_user = [m["content"] for m in history if m["role"] == "user"][-2:]
        if not prev_user:
            return question
        merged = " ".join(prev_user) + " " + question
        return merged.strip()

    # ---------- Retrieval + Self-RAG ----------
    def _reformulate(self, query: str) -> str:
        q = query
        for w in STOPWORDS:
            q = q.replace(w, "")
        return q.strip() or query

    # ---------- Query Decomposition（复杂纠纷分步检索合并） ----------
    def _should_decompose(self, question: str) -> bool:
        """启发式判断是否需查询分解：仅对「复杂、多子问题」的检索类问题开启。

        避免简单问题被无谓拆分（多一次 LLM 调用 + 多次检索，反而变慢变贵）。
        """
        q = question.strip()
        # 调试用：触发条件详情（重启后看 logs/app.log 即可判断为什么没拆）
        log.info("[decompose-debug] q_len=%d qmark=%d qwords=%d | q=%r",
                 len(q), q.count("?") + q.count("？"),
                 q.count("怎么办") + q.count("怎么") + q.count("如何") + q.count("怎样") + q.count("咋"),
                 q[:50])
        if len(q) < 36:
            return False
        # 多个问号
        if q.count("?") + q.count("？") >= 2:
            return True
        # 多个疑问词（怎么/如何/怎么办/怎样/咋）
        qwords = q.count("怎么办") + q.count("怎么") + q.count("如何") + q.count("怎样") + q.count("咋")
        if qwords >= 2:
            return True
        # 并列诉求（连词 + 赔偿/程序/处理等）
        conj = ["还有", "另外", "以及", "同时", "除此之外", "此外", "并且", "加", "和"]
        if any(c in q for c in conj) and any(
            k in q for k in ("赔偿", "怎么办", "怎么", "程序", "处理", "主张", "如何")
        ):
            return True
        return False

    @staticmethod
    def _parse_json_list(raw: str) -> list:
        """从 LLM 输出里稳健地抠出 JSON 数组（兼容代码块 / 多余前后文本）。"""
        if not raw:
            return []
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\[.*\]", raw, re.S)
            if not m:
                return []
            try:
                data = json.loads(m.group(0))
            except Exception:
                return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, (str, int, float))]
        return []

    def _decompose(self, question: str, provider: str, trace_id: str, steps: list) -> list[str]:
        """用 LLM 把复杂问题拆成 2-4 个独立子问题，分别检索后合并。"""
        t = time.perf_counter()
        sys_p = (
            "你是一个社区矛盾调解问题的【查询拆解器】。居民的一段咨询往往混杂多个子问题"
            "（责任认定、举证责任、赔偿范围、解决程序等）。请把用户的问题拆成 2-4 个相互独立、"
            "可分别检索知识库的子问题，每个子问题聚焦于单一维度。\n"
            "只输出一个 JSON 数组，元素为字符串子问题，不要任何解释或 markdown 代码块标记。\n"
            "示例输入：楼上漏水泡坏我天花板装修，邻居不认，物业说过了保修期，我该怎么办、能要多少赔偿、走什么程序？\n"
            "示例输出：[\"楼上漏水导致自家财产受损，法律责任如何认定\"，"
            "\"邻居不承认漏水，举证责任由谁承担、如何取证\"，"
            "\"物业以过保修期为由拒管，其在此事中的责任边界\"，"
            "\"漏水致天花板与装修损坏，可主张哪些赔偿项目及标准\"，"
            "\"邻里漏水纠纷的解决程序：协商、调解、12345、诉讼的先后顺序\"]"
        )
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": question}],
                provider=provider, temperature=0.0,
            )
            subs = self._parse_json_list(raw)
            subs = [str(s).strip() for s in subs if str(s).strip()][: self.s.max_sub_queries]
            if len(subs) < 2:
                return []
            steps.append({"stage": "decompose", "detail": f"子问题={len(subs)}",
                          "ms": round((time.perf_counter() - t) * 1000, 1)})
            return subs
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] 查询分解失败，退回单查询 | %s", trace_id, e)
            return []

    # ---------- 对外接口 ----------
    def query(self, question, provider=None, history=None, session_id=None, stream=False, user_role=None, user_id=None):
        """统一入口。

        stream=False → 返回完整结果 dict（兼容旧调用 / 评测脚本）；
        stream=True  → 返回生成器，逐条 yield SSE 事件
                      {"type": "route"|"delta"|"done"|"error", ...}。
        user_id 用于 B 方案的"用户级长期记忆"：question_log 跨 session 跨角色共享，
        案件档案按 (user_id, role) 隔离。
        """
        gen = self._run(question, provider, history, session_id, user_role=user_role, user_id=user_id)
        if stream:
            return gen
        final = None
        for ev in gen:
            if ev.get("type") == "done":
                final = ev["result"]
        return final

    def _run(self, question, provider, history, session_id, user_role: str | None = None, user_id: str | None = None):
        """内部生成器：逐步执行管道并 yield 事件，供流式输出复用。"""
        import random
        trace_id = uuid.uuid4().hex[:12]
        provider = provider or self.s.default_llm
        steps: list[dict[str, Any]] = []
        t_total = time.perf_counter()
        history = self._normalize_history(history)

        def mark(stage, detail="", start=None):
            ms = (time.perf_counter() - start) * 1000 if start else None
            steps.append({"stage": stage, "detail": detail, "ms": round(ms, 1) if ms is not None else None})

        log.info("[%s] 新请求 | provider=%s | q=%r", trace_id, provider, question[:60])

        # 纯感谢/客套消息短路——在路由之前拦截，不走 LLM（省 token、快、100% 可控）
        _GRATITUDE_KW = ["谢谢", "感谢", "真好", "太好了", "有用", "有帮助",
                         "不错", "厉害", "给力", "棒", "辛苦了"]
        _NOT_PURE = ["但是", "不过", "可是", "还有个", "另外", "接下来", "然后"]
        is_pure_thanks = (
            any(kw in question for kw in _GRATITUDE_KW)
            and len(question.strip()) < 30
            and not any(x in question for x in _NOT_PURE)
        )
        if is_pure_thanks:
            replies = [
                "不客气，有需要随时问。",
                "能帮到你就好，有问题再找我。",
                "不客气，希望对你有帮助！",
                "应该的，有问题随时说。",
            ]
            mark("supervise", "route=gratitude-shortcut")
            yield {"type": "route", "route": "direct", "trace_id": trace_id, "session_id": session_id}
            wrapped = self._wrap(trace_id, steps, t_total, "direct", random.choice(replies), [], 0, provider)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return

        # Supervisor 路由（有历史时，短追答不再误判为 clarify）
        t = time.perf_counter()
        route = self._supervise(question, has_history=bool(history))
        mark("supervise", f"route={route}", t)
        log.info("[%s] 路由判定=%s | 历史轮数=%d", trace_id, route, len(history))

        if route == "direct":
            yield {"type": "route", "route": "direct", "trace_id": trace_id, "session_id": session_id}
            wrapped = self._wrap(trace_id, steps, t_total, "direct", self._direct_answer(), [], 0, provider)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return
        if route == "clarify":
            yield {"type": "route", "route": "clarify", "trace_id": trace_id, "session_id": session_id}
            wrapped = self._wrap(trace_id, steps, t_total, "clarify", self._clarify(), [], 0, provider)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return
        if route == "out_of_domain":
            log.info("[%s] 超出服务范围，直接回复（不检索）", trace_id)
            yield {"type": "route", "route": "out_of_domain", "trace_id": trace_id, "session_id": session_id}
            wrapped = self._wrap(trace_id, steps, t_total, "out_of_domain", self._out_of_domain(), [], 0, provider)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return

        # ---------- Query Decomposition（复杂纠纷分步检索合并）----------
        decomposition: dict[str, Any] = {"enabled": False, "sub_queries": []}
        retries = 0
        is_mock = self.embedder.use_mock
        thr = 0.0 if is_mock else self.s.relevance_threshold

        if (not is_mock) and self.s.enable_decomposition and self._should_decompose(question):
            sub_queries = self._decompose(question, provider, trace_id, steps)
            if sub_queries:
                all_hits: list[dict] = []
                seen_ids: set = set()
                for sq in sub_queries:
                    sq_vec = self.embedder.embed_query(sq)
                    hits, _ = self._retrieve(sq, trace_id, steps, query_vec=sq_vec)
                    for h in hits:
                        hid = h.get("id")
                        if hid and hid not in seen_ids:
                            seen_ids.add(hid)
                            all_hits.append(h)
                if all_hits:
                    all_hits.sort(
                        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
                        reverse=True,
                    )
                    ranked = all_hits[: self.s.top_k * 2]
                    best = ranked[0].get("rerank_score", ranked[0].get("score", 0.0))
                    decomposition = {"enabled": True, "sub_queries": sub_queries}
                    log.info("[%s] 查询分解完成 | 子问题=%d 合并去重后命中=%d", trace_id, len(sub_queries), len(all_hits))
                else:
                    sub_queries = []  # 子检索无果，退回单查询路径
            if not decomposition["enabled"]:
                # 退回普通单查询 + Self-RAG 改写重试
                query = self._contextual_query(question, history)
                if query != question:
                    log.info("[%s] 上下文合并检索 | %r -> %r", trace_id, question, query)
                    steps.append({"stage": "context_merge", "detail": f"{question!r} -> {query!r}", "ms": None})
                qvec = self.embedder.embed_query(query)
                ranked, best = self._retrieve(query, trace_id, steps, query_vec=qvec)
                while best < thr and retries < self.s.max_retrieve_retries:
                    new_query = self._reformulate(query)
                    log.info("[%s] 低于阈值(%.4f<%.4f) 改写查询 | %r -> %r", trace_id, best, thr, query, new_query)
                    steps.append({"stage": f"reformulate_r{retries + 1}", "detail": f"{query!r} -> {new_query!r}", "ms": None})
                    query = new_query
                    ranked, best = self._retrieve(query, trace_id, steps, retry=retries + 1)
                    retries += 1
        else:
            # 简单问题 / mock 模式：普通单查询 + Self-RAG 改写重试
            query = self._contextual_query(question, history)
            if query != question:
                log.info("[%s] 上下文合并检索 | %r -> %r", trace_id, question, query)
                steps.append({"stage": "context_merge", "detail": f"{question!r} -> {query!r}", "ms": None})
            qvec = self.embedder.embed_query(query)
            ranked, best = self._retrieve(query, trace_id, steps, query_vec=qvec)
            while best < thr and retries < self.s.max_retrieve_retries:
                new_query = self._reformulate(query)
                log.info("[%s] 低于阈值(%.4f<%.4f) 改写查询 | %r -> %r", trace_id, best, thr, query, new_query)
                steps.append({"stage": f"reformulate_r{retries + 1}", "detail": f"{query!r} -> {new_query!r}", "ms": None})
                query = new_query
                ranked, best = self._retrieve(query, trace_id, steps, retry=retries + 1)
                retries += 1

        if not ranked or best < thr:
            honest = (
                "抱歉，知识库中暂未检索到与您描述情形直接对应的调解依据。"
                "建议补充矛盾类型与关键事实，或联系社区调解委员会获取人工协助。"
            )
            log.warning("[%s] 诚实拒答 | 最佳相关度=%.4f < 阈值=%.4f | 重试=%d", trace_id, best, thr, retries)
            yield {"type": "route", "route": "retrieve", "trace_id": trace_id, "session_id": session_id}
            wrapped = self._wrap(trace_id, steps, t_total, "retrieve", honest, [], retries, provider)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return

        role = self._resolve_role(user_role, question, history)
        mark("infer_role", f"role={role}")
        # B 方案：案件档案按 (user_id, role) 隔离 —— 同一角色跨 session 共享专业记忆，跨角色不串
        # 兼容旧逻辑：当 user_id 缺失时回退到原 session_id 键
        if user_id:
            case_profile = self._user_mem.get_profile(user_id, role)
        else:
            case_profile = self._profiles.get(session_id) if session_id else None

        # B 方案：检测"回忆类"问题 → 跳过 RAG，直接基于用户级 question_log 用 LLM 总结
        # 例：「我刚才问了什么」「之前/上次问过什么」「我最近问过什么」
        recall = self._maybe_recall(question, user_id, role) if user_id else None
        if recall:
            t_re = time.perf_counter()
            mark("recall_match", "命中回忆关键词 → 跳过 RAG，直接基于 question_log 生成")
            yield {"type": "route", "route": "recall", "trace_id": trace_id, "session_id": session_id,
                   "steps": list(steps)}
            answer_parts: list[str] = []
            for delta in self._generate_recall_answer(question, recall, provider):
                answer_parts.append(delta)
                yield {"type": "delta", "text": delta}
            answer = "".join(answer_parts).strip()
            steps.append({"stage": "generate", "detail": f"mode=recall 字数={len(answer)}",
                          "ms": round((time.perf_counter() - t_re) * 1000, 1)})
            wrapped = self._wrap(trace_id, steps, t_total, "recall", answer, [], 0, provider,
                                 user_role=role, case_profile=None, decomposition=None)
            wrapped["session_id"] = session_id
            yield {"type": "done", "result": wrapped}
            return

        # 检索链路已就绪，先把「路由 + 已完成步骤」推给前端（首字前可见骨架）
        yield {"type": "route", "route": "retrieve", "trace_id": trace_id, "session_id": session_id,
               "steps": list(steps)}

        # 流式生成答案（首字即可见，体感远快于等整段）
        answer_parts: list[str] = []
        t_gen = time.perf_counter()
        for delta in self._generate_stream(question, ranked, provider, trace_id, role=role, history=history, profile=case_profile, decomposed=decomposition["enabled"], recall=recall):
            answer_parts.append(delta)
            yield {"type": "delta", "text": delta}
        answer = "".join(answer_parts)
        # 兜底清理：清除模型偶发复述的提示词标记
        answer = answer.replace("【参考依据】", "")
        answer = re.sub(r"^\s*参考依据[：:].*$", "", answer, flags=re.M)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        steps.append({"stage": "generate", "detail": f"model={provider} 字数={len(answer)}",
                      "ms": round((time.perf_counter() - t_gen) * 1000, 1)})

        # 本轮结束后更新并持久化案件档案（合并累积，不丢历史事实）
        # B 方案：按 (user_id, role) 键保存，使同一角色跨 session 共享专业记忆
        if user_id:
            existing = case_profile or {}
            fresh = self._build_case_profile(history, question, role)
            merged = self._merge_profile(existing, fresh)
            self._user_mem.save_profile(user_id, role, merged)
            case_profile = merged
        elif session_id:
            existing = case_profile or {}
            fresh = self._build_case_profile(history, question, role)
            merged = self._merge_profile(existing, fresh)
            self._profiles.save(session_id, merged)
            case_profile = merged

        log.info("[%s] 完成 | 路由=retrieve 来源数=%d 重试=%d 角色=%s", trace_id, len(ranked), retries, role)
        wrapped = self._wrap(trace_id, steps, t_total, "retrieve", answer, ranked, retries, provider,
                             user_role=role, case_profile=case_profile, decomposition=decomposition)
        wrapped["session_id"] = session_id
        yield {"type": "done", "result": wrapped}

    def _retrieve(self, query: str, trace_id: str, steps: list, retry: int = 0, query_vec: list[float] | None = None) -> tuple[list[dict], float]:
        t = time.perf_counter()
        vec = query_vec if query_vec is not None else self.embedder.embed_query(query)
        dense = self.store.search(vec, top_k=self.s.top_k)

        # 混合检索：稠密 ∪ 稀疏(BM25) → RRF 融合扩大候选池 → rerank 精排
        if self.s.enable_hybrid and self.bm25.is_built:
            sparse = self.bm25.search(query, top_k=self.s.top_k)
            fused = rrf_fuse([dense, sparse], k=self.s.rrf_k)
            # 把 dense 余弦分回填到融合候选，供 rerank 的 hybrid_score 使用。
            # 对 BM25 独有命中（向量未召回）的文档，用 dense 最低分作为估计值，
            # 避免 score=0 被 rerank 的 0.7×向量权重直接压死，保证 RRF 纠错不被抹平。
            dense_scores = {d["id"]: d["score"] for d in dense}
            min_dense = min((d["score"] for d in dense), default=0.0)
            for f in fused:
                f["score"] = dense_scores.get(f["id"], min_dense)
            candidates = fused
            mode = "hybrid"
        else:
            candidates = dense
            mode = "vector"

        ranked = rerank(query, candidates, top_k=self.s.rerank_top_k)
        best = ranked[0]["rerank_score"] if ranked else 0.0
        label = f"retry={retry} " if retry else ""
        log.info(
            "[%s] 检索%s[%s] | 候选=%d 命中top=%d 最佳分=%.4f",
            trace_id, label, mode, len(candidates), len(ranked), best,
        )
        mark = "retrieve"
        if retry:
            mark = f"retrieve_r{retry}"
        steps.append({
            "stage": mark,
            "detail": f"mode={mode} 候选={len(candidates)} top={len(ranked)} best={best:.4f}",
            "ms": round((time.perf_counter() - t) * 1000, 1),
        })
        return ranked, best

    # ---------- 用户角色（由前端身份选择器指定，不再猜测） ----------
    VALID_ROLES = {"resident", "mediator", "property", "grid_worker"}

    def _resolve_role(self, explicit_role: str | None, question: str, history: list[dict] | None = None) -> str:
        """确定用户角色：优先用前端传入的显式角色，否则降级为关键词推断。

        前端在首次对话时引导用户选择身份（居民/物业/调解员/网格员），
        选定后每条请求都带 user_role 字段。只有当该字段为空时才走关键词推断。
        """
        if explicit_role and explicit_role in self.VALID_ROLES:
            return explicit_role
        # 降级推断
        return self._infer_role(question, history)

    def _infer_role(self, question: str, history: list[dict] | None = None) -> str:
        """从用户措辞推断其身份，用于决定答案视角。

        默认 'resident'
        命中调解/社区工作口吻则判定为 'mediator'；命中物业职责口吻为 'property'。
        多轮场景下把历史用户话合并判断，避免碎片化追答丢失身份。
        """
        q = question
        if history:
            q = " ".join([m["content"] for m in history if m["role"] == "user"]) + " " + question
        if any(k in q for k in [
            "接案", "接到投诉", "受理登记", "如何调解", "怎么调解", "调解流程",
            "组织座谈", "上门走访", "回访", "调处", "社区工作站", "网格员", "网格",
        ]):
            return "mediator"
        if any(k in q for k in [
            "物业怎么", "作为物业", "物业如何", "管家", "巡查记录", "物业上报", "工程维修单",
        ]):
            return "property"
        # 居民 / 当事人 / 投诉人（含显式自述或默认）
        return "resident"

    def _role_guidance(self, role: str, known_facts: dict[str, str] | None = None) -> str:
        """按角色返回生成约束，嵌进 _generate 的提示词。"""
        known = known_facts or {}
        known_dims = "、".join(known.keys()) if known else ""
        if role == "mediator":
            return (
                "【对话对象】社区调解员 / 社工 / 居委会工作人员。\n"
                "【回答要求】按专业流程组织：受理登记→核实走访→组织调解→签订约定→回访。"
                "直接引用知识库步骤原文，专业口吻，不废话不客套。\n"
            )
        if role == "property":
            return (
                "【对话对象】物业服务人员（管家/工程/安保等）。\n"
                "【回答要求】从物业职责角度给可执行动作：现场核实、台账记录、协调工程/安保、"
                "向业主反馈、上报社区。物业工作口吻，简洁直接。\n"
            )
        if role == "grid_worker":
            return (
                "【对话对象】网格员 / 社工（一线走访人员）。\n"
                "【回答要求】从一线走访视角给行动指引：上门安全注意、证据固定、联动资源（"
                "社区/派出所/民政）、文书模板。务实口吻，关注执行细节。\n"
            )
        # resident（默认）
        ask_rule = (
            f"3) 【默认给建议】基于已有信息直接给出 2-4 条可执行建议。"
            f"只在信息确实严重不足时，才可在建议末尾简短提一句还需要什么信息。"
            f"【严禁追问以下用户已告知的维度】：{known_dims}。"
            f"绝对不要每轮都追问，不要当面试官。"
            if known_dims else
            "3) 【默认给建议】基于已有信息直接给出 2-4 条可执行建议。"
            "不要每轮都追问用户问题——用户来这里是求解决方案的，不是来接受面试的。"
            "只在信息确实严重不足时，才可在建议末尾简短提一句还需要什么信息。"
        )
        return (
            "【对话对象】居民 / 业主 / 当事人（遇到矛盾的一方）。\n"
            "【回答要求】\n"
            "1) 不要任何客套话（禁止「别急」「理解你的困扰」「抱歉听到这些」），直接给建议。\n"
            "2) 站在『你能做什么』的角度。知识库很多是从调解员视角写的，必须转换为对居民的行动建议"
            "（例：『调解员应上门走访』→『你可以请物业或社区上门核实，自己同时留存录音/视频』）；"
            "绝不能把『调解员要做的事』当成『你要做的事』。\n"
            f"{ask_rule}\n"
            "4) 一次只给最紧急、最可执行的 2-4 条，每条一句话带过，不要展开解释『为什么这么做』。\n"
            "5) 法条用大白话解释，不要念原文。\n"
        )

    # ---------- 已知事实提取（防止重复追问已回答的信息） ----------
    # 常见关键事实维度 + 对应的口语关键词（覆盖居民常用表达方式）
    _FACT_DIMENSIONS = {
        "起止时间": ["几点", "几点钟", "早上", "中午", "下午", "晚上", "凌晨", "半夜",
                    "深夜", "从.*点", "到.*点", "开始.*搞", "搞到", "持续到",
                    "一直搞到", "搞到.*才", "从.*开始", "到.*结束", "到.*才停",
                    "上午", "傍晚", "夜里", "通宵", "整晚"],
        "持续时间/频率": ["持续", "多久", "几天", "几个月", "好久了", "一直", "经常",
                     "天天", "每天", "每晚", "偶尔", "一次", "频率", "频次",
                     "又开始了", "还是照样", "照旧", "依旧"],
        "是否已沟通": ["沟通过", "找过他", "说过", "跟他讲", "找过楼上", "找过对方",
                      "找过邻居", "反映过", "跟他说了", "交涉", "协商", "找过.*毛"],
        "沟通结果/对方态度": ["不改", "不听", "不理", "骂回来", "态度差", "不认",
                             "推脱", "敷衍", "答应但没做", "口头答应", "没用", "无效",
                             "照样", "没改", "照旧", "依旧", "还是照样", "还是不改"],
        "是否找过物业/社区": ["物业", "管家", "管理处", "居委会", "社区", "调解员",
                            "报警", "派出所", "110", "12345", "街道"],
        "证据情况": ["录音", "录像", "视频", "拍照", "截图", "聊天记录", "微信",
                    "留证", "证据", "拍下来", "录下来"],
        "影响程度": ["睡不着", "睡不好", "影响休息", "影响学习", "小孩", "孩子",
                    "老人", "病人", "神经衰弱", "精神", "质量差"],
    }

    def _extract_known_facts(self, history: list[dict] | None, current_question: str = "") -> dict[str, str]:
        """扫描对话历史 + 当前用户输入，提取已透露的关键事实。

        返回 {维度: 摘要}，用于注入提示词，让 LLM 明确知道「什么已经知道了、不要再问」。
        """
        # 构造完整扫描列表：历史 + 当前问题（当前轮次的信息也要纳入）
        all_user_msgs = list(history) if history else []
        if current_question and len(current_question.strip()) >= 2:
            all_user_msgs.append({"role": "user", "content": current_question})
        if not all_user_msgs:
            return {}
        facts: dict[str, str] = {}
        # 逐轮扫描用户发言（保留自然句边界），从最近往前找
        # 特殊处理：起止时间维度需要" richest match"——
        #   "从9点开始搞到凌晨" >> "天天晚上搞"，所以不能首匹配即停。
        _TIME_RICH_KEYWORDS = ("从.*点", "到.*点", "开始.*搞", "搞到", "持续到",
                               "一直搞到", "\\d+点", "\\d+:\\d+", "整晚", "通宵", "彻夜")
        time_candidates: list[str] = []  # 收集所有起止时间候选，最后选最丰富的

        for m in reversed(all_user_msgs):
            if m["role"] not in ("user",):
                continue
            text = m["content"]
            # 按中英文句号/感叹号/问号/逗号/空格/换行分句（覆盖无标点口语）
            sentences = re.split(r"[。！？.!?,，\s\n]+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 4:
                    continue
                for dim, keywords in self._FACT_DIMENSIONS.items():
                    if dim == "起止时间":
                        # 起止时间：收集所有候选不截断，后面统一选最优
                        for kw in keywords:
                            match = re.search(kw, sent)
                            if match:
                                idx = match.start()
                                start = max(0, idx - 10)
                                end = min(len(sent), idx + len(match.group(0)) + 30)
                                candidate = sent[start:end].strip()
                                if len(candidate) >= 4:
                                    time_candidates.append(candidate)
                        continue
                    if dim in facts:
                        continue  # 该维度已找到，跳过
                    for kw in keywords:
                        # 支持正则模式（如 "从.*点"）和纯子串（如 "天天"）
                        match = re.search(kw, sent)
                        if match:
                            # 截取匹配位置前后各 20 字符的窗口作为摘要（保持上下文）
                            idx = match.start()
                            start = max(0, idx - 20)
                            end = min(len(sent), idx + len(match.group(0)) + 40)
                            snippet = sent[start:end].strip()
                            # 窗口过短（如口语短句被空格切开）时直接用整句
                            if len(snippet) < 4:
                                snippet = sent
                            if len(snippet) >= 4:
                                facts[dim] = snippet[:80]
                            break
            # 所有非时间维度都找到就可提前结束（起止时间单独处理）
            non_time_found = sum(1 for d in facts if d != "起止时间")
            if non_time_found >= len(self._FACT_DIMENSIONS) - 1:
                break

        # === 起止时间：从所有候选中选信息量最丰富的一条 ===
        def _rich_score(s: str) -> int:
            score = len(s)
            if re.search(r"\d+[点:时：]", s): score += 50   # 含具体时刻
            if any(w in s for w in ("凌晨", "半夜", "深夜", "通宵", "彻夜")): score += 40
            if any(w in s for w in ("从", "到", "开始", "结束", "持续")): score += 20
            return score

        if time_candidates:
            time_candidates.sort(key=_rich_score, reverse=True)
            facts["起止时间"] = time_candidates[0][:80]

        # === 补充扫描：起止时间经常跨逗号分句（如"从9点开始就搞，搞到凌晨"），
        #     上述逐句扫描可能只抓到一半。这里对最新一条用户发言做整段正则补捞。 ===
        if all_user_msgs:
            latest = ""
            for m in reversed(all_user_msgs):
                if m.get("role") == "user":
                    latest = m.get("content", "")
                    break
            if latest:
                # 匹配 "从X点...到/搞到/持续到...Y点/凌晨/半夜" 等时间范围表达
                time_patterns = [
                    r"从.{0,6}点.{0,10}(到|搞到|持续到|一直).{0,6}(点|凌晨|半夜|深夜|早上|上午|下午|傍晚|夜里|通宵)",
                    r".{0,4}点.{0,6}(开始|就开始).{0,15}(搞到|到|持续到).{0,6}(点|凌晨|半夜|深夜)",
                    r"(早上|中午|下午|晚上|傍晚|凌晨|半夜|深夜).{0,8}(开始|就).{0,15}(搞|弄|吵|响).{0,10}(到|持续|一直到).{0,8}(点|凌晨|半夜|深夜|早上)",
                    r"每天?从.{0,6}(点|左右).{0,15}(到|搞到|持续到).{0,6}(点|凌晨|半夜)",
                    r"\d{1,2}[点:时]\D{0,10}(到|—|~|至|搞到|持续到)\D{0,10}\d{1,2}[点:时]",
                    r"(整晚|通宵|彻夜|一整夜|全天|从早到晚)",
                ]
                for pat in time_patterns:
                    m = re.search(pat, latest)
                    if m:
                        snippet = m.group(0)[:80]
                        if len(snippet) >= 4:
                            # 只有当正则补捞的结果比已有候选更丰富时才升级
                            current_score = _rich_score(facts.get("起止时间", ""))
                            regex_score = _rich_score(snippet)
                            if regex_score > current_score:
                                facts["起止时间"] = snippet
                        break

        return facts

    # ---------- AI 已问问题追踪（防止 AI 重复问同一个问题） ----------
    _QUESTION_PATTERNS = [
        r"有没有.*记录|有没有.*录音|有没有.*录像|有没有.*截图|有没有.*证据|有没有.*照片",
        r"持续.*多久|多久了|几天.*了|什么时候开始的|几点.*开始|持续到",
        r"沟通过.*没|找过.*没|跟.*说过.*没|有没有.*沟通|有没有.*交涉|有没有.*协商",
        r"对方.*什么态度|他.*怎么回应|他.*怎么说|对方.*反应",
        r"物业.*没|社区.*没|居委会.*没|报警.*没|派出所.*没|12345.*没|街道.*没|管家.*没",
        r"影响.*怎样|影响.*如何|睡得好不好|休息.*影响|学习.*影响|工作.*影响",
    ]

    def _extract_asked_questions(self, history: list[dict] | None) -> list[str]:
        """从 AI 自己的历史回答中提取已经问过用户的问题摘要。

        解决的问题是：AI 在第2轮问了「有没有留下录音」，第4轮又问完全一样的问题。
        _extract_known_facts 只能追踪用户说过的信息，但无法阻止 AI 重复自己的提问。
        这里扫描 assistant 历史消息，用正则匹配追问句式，返回已问过的问题列表。
        """
        if not history:
            return []
        asked: list[str] = []
        seen: set[str] = set()
        for m in history:
            if m.get("role") not in ("assistant", "bot"):
                continue
            text = m.get("content", "")
            for pat in self._QUESTION_PATTERNS:
                match = re.search(pat, text)
                if match:
                    q = match.group(0).strip()
                    # 归一化：去掉开头通用词便于去重
                    key = re.sub(r"^[有没有是否]", "", q).strip()[:30]
                    if key not in seen and len(q) >= 6:
                        seen.add(key)
                        asked.append(q)
        return asked

    # ---------- B 方案：用户级 question_log 回忆注入 ----------
    # 「我刚才问了什么」「之前/上次问过什么」「我最近问过什么」类问题
    # → 拉取用户级最近提问索引，注入 prompt 让 AI 据此作答
    _RECALL_KEYWORDS = [
        "刚才问了什么", "刚才问过", "刚才提过", "刚才说",
        "之前问了", "之前问过", "之前提过",
        "上次问了", "上次问过", "上次提过",
        "最近问了", "最近问过", "最近提过",
        "我问过什么", "我问过啥", "我问过哪些",
        "我之前问了", "我刚才问", "我之前问",
        "我最近问", "我上轮问", "我前面问",
        "我提过什么", "我提过啥", "我说过什么",
        "有没有问过", "问过什么", "问过哪些",
        "记得我问过", "记得我", "我们聊过", "之前聊过",
    ]

    def _maybe_recall(self, question: str, user_id: str, role: str) -> str | None:
        """检测「回忆类」问题并返回要注入 prompt 的块（无命中返回 None）。

        实现要点：
        1. 关键词匹配判定（轻量、不调 LLM）
        2. 跨 session 跨角色取最近 10 条（role 维度与当前不一致时同时标注）
        3. 提示 AI 必须基于【用户最近问过】块作答，引用时可指明角色/时间
        """
        if not user_id:
            return None
        q = (question or "").strip()
        if not q or len(q) > 60:
            # 回忆类问题通常很短（"我刚才问了什么" 类）；过长不算
            return None
        hit = any(kw in q for kw in self._RECALL_KEYWORDS)
        if not hit:
            return None
        items = self._user_mem.get_recent_questions(user_id, role=None, n=10)
        if not items:
            return (
                "【用户最近问过】\n"
                "（这是该用户本轮会话之外的提问记录，目前还没有任何记录。）\n"
                "请直接告诉用户：暂时还没有跨会话的提问记录。"
            )
        lines: list[str] = []
        for it in items:
            ts = it.get("ts") or 0
            r = it.get("role") or "未指定"
            preview = it.get("preview") or it.get("question") or ""
            try:
                import datetime as _dt
                tstr = _dt.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
            except Exception:
                tstr = "时间未知"
            lines.append(f"  - [{tstr}] 以【{r}】身份：{preview}")
        block = (
            "【用户最近问过（跨 session 跨角色汇总，按时间倒序，最多 10 条）】\n"
            + "\n".join(lines)
            + "\n"
            "请基于以上记录直接告诉用户他/她之前问过什么；"
            "如用户限定了角色视角，优先呈现该角色的记录，并在引用时简短说明来源（时间/角色）。"
        )
        return block

    def _generate_recall_answer(self, question: str, recall_block: str, provider: str):
        """回忆类问题的生成器：跳过 RAG，直接基于 question_log 用 LLM 总结回答。

        失败时回退到 question_log 模板化摘要，保证用户至少看到记录原文。
        """
        system = (
            "你是社区矛盾调解助理。用户问的是「我之前/刚才问过什么」类问题。"
            "请严格基于下方【用户最近问过】记录，用 2-5 句自然语言告诉用户他/她最近问过什么；"
            "可以按时间倒序列出 1-3 条要点（每条一句话概括），并简短注明对应角色/时间；"
            "如果记录显示「目前还没有任何记录」，直接告诉用户暂未记录到他/她的提问即可。"
            "严禁编造记录中不存在的内容；如对记录不明确，承认即可。"
        )
        user = f"用户问题：{question}\n\n{recall_block}"
        # 模板回退（mock / LLM 失败时使用）
        def _fallback():
            return (
                "根据我的记忆，"
                + recall_block.replace("【用户最近问过（跨 session 跨角色汇总，按时间倒序，最多 10 条）】\n", "")
                .replace("（这是该用户本轮会话之外的提问记录，目前还没有任何记录。）\n请直接告诉用户：暂时还没有跨会话的提问记录。",
                         "暂未记录到您之前的提问。")
            )
        if self.s.mock:
            yield _fallback()
            return
        try:
            buf: list[str] = []
            for delta in self.llm.stream_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                provider=provider, temperature=0.2,
            ):
                buf.append(delta)
                yield delta
            if not "".join(buf).strip():
                yield _fallback()
        except Exception as e:  # noqa: BLE001
            log.warning("回忆回答 LLM 失败，回退模板: %s", e)
            yield _fallback()

    # ---------- 长期记忆（P1）：结构化案件档案 ----------
    _CASE_TYPE_MAP = {
        "噪音纠纷": ["噪音", "噪声", "装修", "扰民", "嗡嗡", "施工", "吵"],
        "漏水/渗水": ["漏水", "渗水", "水管", "积水", "滴水"],
        "停车纠纷": ["停车", "车位", "地锁", "充电桩"],
        "宠物纠纷": ["宠物", "狗", "猫", "犬", "畜生"],
        "物业纠纷": ["物业", "物业费", "维修基金", "业委会"],
        "违建纠纷": ["违建", "搭建", "绿地", "采光", "通风", "加盖"],
        "油烟/环境": ["油烟", "垃圾", "环境", "排污", "排水", "路灯", "污染"],
        "租赁纠纷": ["房东", "租客", "租户", "出租", "群租"],
        "家事纠纷": ["赡养", "抚养", "家暴", "家庭暴力", "离婚"],
    }
    _OPPONENT_TOKENS = ["楼上邻居", "楼上", "楼下邻居", "楼下", "邻居", "物业",
                       "房东", "租客", "租户", "业委会", "居委会", "开发商"]

    def _build_case_profile(self, history: list[dict] | None, question: str, role: str) -> dict:
        """从对话（历史 + 当前问题）规则化抽取结构化案件档案。

        与 _extract_known_facts（只服务单轮防重复追问）不同，这里产出的档案会被
        持久化到 Redis，跨多轮、跨上下文窗口截断后仍是稳定的「事实源」——
        即使原始多轮文本被截断，AI 也能依据档案作答，而不是凭空编造。

        注意：档案只反映【用户说过的话】，绝不把 AI 的回答内容算进去
        （否则 AI 建议「可以报警/录音」会被误读为用户已报警、已取证）。
        """
        texts: list[str] = []
        if question:
            texts.append(question)
        for m in (history or []):
            # 只采集用户发言，排除 assistant（AI 的自身建议不算用户事实）
            if m.get("role") == "user" and m.get("content"):
                texts.append(m["content"])
        full = "\n".join(texts)
        low = full.lower()

        # 案件类型（命中即归类，多个命中取首个）
        case_type = ""
        for ct, kws in self._CASE_TYPE_MAP.items():
            if any(k in low for k in kws):
                case_type = ct
                break

        # 对方当事人（最长优先匹配）
        opponent = ""
        for tok in sorted(self._OPPONENT_TOKENS, key=len, reverse=True):
            if tok in full:
                opponent = tok
                break

        # 证据情况
        ev_kw = ["录音", "录像", "视频", "照片", "截图", "聊天记录", "证据", "报警记录", "物业记录"]
        has_ev = [k for k in ev_kw if k in full]
        evidence_status = "已留存：" + "、".join(has_ev) if has_ev else "未提及/未留存"

        # 已知事实（复用已有的维度抽取）
        key_facts = self._extract_known_facts(history, question)

        # 当前阶段（启发式，从行政/司法程序往下覆盖）
        if any(k in full for k in ["报警", "派出所", "城管", "法院", "起诉", "诉讼", "律师"]):
            stage = "已进入行政/司法程序"
        elif any(k in full for k in ["物业介入", "物业协调", "居委会介入", "调解委员会", "调解员", "社区调解"]):
            stage = "已申请社区调解"
        elif any(k in full for k in ["录像", "录音", "取证", "留证", "证据"]):
            stage = "证据固定阶段"
        elif any(k in full for k in ["沟通过", "说过", "反映过", "找过他", "交涉", "协商"]):
            stage = "已自行沟通"
        else:
            stage = "初步咨询"

        return {
            "identity": role,
            "case_type": case_type,
            "opponent": opponent,
            "evidence_status": evidence_status,
            "stage": stage,
            "key_facts": key_facts,
            "updated_at": time.time(),
        }

    def _merge_profile(self, old: dict, fresh: dict) -> dict:
        """累积式合并：已知事实只增不减，结构化字段新值优先、空则保留旧值。"""
        merged = dict(fresh)
        if old:
            kf = dict(old.get("key_facts", {}))
            kf.update(fresh.get("key_facts", {}))
            merged["key_facts"] = kf
            for f in ("case_type", "opponent", "evidence_status", "stage", "identity"):
                if not merged.get(f) and old.get(f):
                    merged[f] = old[f]
        merged["updated_at"] = time.time()
        return merged

    def _generate_stream(self, question: str, sources: list[dict], provider: str, trace_id: str, role: str = "resident", history: list[dict] | None = None, profile: dict | None = None, decomposed: bool = False, recall: str | None = None):
        """流式生成：逐块 yield 文本增量（mock 模式下整段一次性 yield）。

        纯感谢短路已上移到 _run，入口保证不会传入纯客套消息。
        recall：B 方案的用户级 question_log 注入块（回忆类问题才非空）
        """
        ctx_blocks = []
        for i, s in enumerate(sources, 1):
            p = s["payload"]
            step_text = "; ".join(p.get("mediation_steps", []))
            ctx_blocks.append(
                f"[{i}]（编号 {p.get('id')}｜{p.get('category')}）\n"
                f"标题：{p.get('title')}\n内容：{p.get('content')}\n"
                f"法条：{p.get('legal_basis')}\n步骤：{step_text}"
            )
        context = "\n\n".join(ctx_blocks)
        # v2：严格接地提示词——只依据检索资料，不补未见于资料的法条/事实，
        # 资料不足即诚实拒答。这是提升 RAGAS faithfulness 的关键改动。
        # v3：叠加【用户角色感知】——根据对话对象身份切换答案视角，
        # 把知识库（多为调解员操作手册视角）转换为对应用户角色的可执行建议。
        # v4：叠加【已知事实感知】——从历史提取用户已回答的关键信息，
        # 禁止重复追问已知事项（这是导致用户体验差的核心原因）。
        # v5：叠加【AI 已问问题追踪】——从 AI 自己的历史回答中提取已问过的问题，
        # 防止 AI 在不同轮次重复问同一个问题（如反复问"有没有录音"）。
        known = self._extract_known_facts(history, question)
        asked = self._extract_asked_questions(history)
        role_label = {"resident": "居民/业主（当事人）", "mediator": "调解员/居委会", "property": "物业服务人员", "grid_worker": "网格员/社工"}[role]
        role_g = self._role_guidance(role, known)
        fact_block = ""
        if known:
            fact_lines = "\n".join(f"  - {k}：{v}" for k, v in known.items())
            extra_warning = ""
            if "起止时间" in known:
                extra_warning = (
                    "\n⚠️ 特别注意：用户已经明确说了具体的起止时间（如几点到几点），"
                    "你绝对不能再问「几点开始」「持续到几点」「具体时间段」这类问题！"
                    "这会让用户非常愤怒，觉得你根本没在听。直接用已知的时间信息给建议即可。\n"
                )
            fact_block = (
                f"\n【用户已在对话中透露的信息（绝对不要再问这些）】\n"
                f"{fact_lines}\n"
                f"以上信息用户已经告诉你了，直接基于这些已知条件给下一步建议，"
                f"严禁以任何形式再次询问上述已知的维度。{extra_warning}\n"
            )
        asked_block = ""
        if asked:
            asked_lines = "\n".join(f"  - {q}" for q in asked)
            asked_block = (
                f"\n【你在之前的回答中已经问过的问题（绝对不要再问同样的内容）】\n"
                f"{asked_lines}\n"
                f"以上问题你已经在前面问过用户了，重复提问会让用户觉得你不专业、"
                f"没有在听。直接基于已有信息给建议，或问一个全新的、从未问过的角度。\n"
            )
        # B 方案：用户级 question_log 回忆块（跨 session/跨角色汇总）
        recall_block = ""
        if recall:
            recall_block = f"\n{recall}\n"
        # 长期记忆（P1）：把已沉淀的结构化案件档案作为「事实源」注入，
        # 即使原始多轮对话被上下文窗口截断，AI 仍依据真实事实作答，不瞎编。
        profile_block = ""
        if profile:
            plines: list[str] = []
            if profile.get("case_type"):
                plines.append(f"案件类型：{profile['case_type']}")
            if profile.get("opponent"):
                plines.append(f"对方当事人：{profile['opponent']}")
            if profile.get("stage"):
                plines.append(f"当前阶段：{profile['stage']}")
            if profile.get("evidence_status"):
                plines.append(f"证据情况：{profile['evidence_status']}")
            for k, v in profile.get("key_facts", {}).items():
                plines.append(f"已知事实【{k}】：{v}")
            if plines:
                profile_block = (
                    "\n【本次会话已沉淀的案件档案（结构化事实源，直接采用，严禁与之矛盾或编造）】\n"
                    + "\n".join(f"  - {l}" for l in plines) + "\n"
                    "以上是你在本会话中已掌握的客观事实，回答必须与之保持一致；"
                    "若用户本轮给出了与档案冲突的新说法，以用户最新表述为准，"
                    "但不得在回答中声称档案里没有的信息，也不要把用户没说过的内容脑补进去。\n"
                )
        # v6：检测用户表达感谢/满意/正面评价 → 强制短回应模式
        _GRATITUDE_PATTERNS = [
            "谢谢", "感谢", "真好", "太好了", "有用", "有帮助", "不错",
            "厉害", "给力", "棒", "辛苦了", "好的谢谢", "谢谢你",
        ]
        is_gratitude = any(p in question for p in _GRATITUDE_PATTERNS) and len(question) < 30
        gratitude_override = ""
        if is_gratitude:
            gratitude_override = (
                "\n【重要】用户正在表达感谢或满意。你的回答必须满足：\n"
                "1) 全文不超过 2 句话，总字数不超过 50 字；\n"
                "2) 只做自然简短的回应（如「不客气，有需要随时问」「能帮到你就好」）；\n"
                "3) 绝对禁止出现以下内容：自我介绍、产品定位、功能说明、"
                "\"我是xx助理\"、\"专注于xx场景\"、\"结合知识库」等任何形式的背书；\n"
                "4) 不要提资料、法条、检索结果。\n\n"
            )

        # Query Decomposition 命中：引导按子问题维度分点覆盖
        decomp_guidance = ""
        if decomposed:
            decomp_guidance = (
                "6) 用户问题已被拆分为多个子问题并分别检索合并了资料；"
                "请按子问题维度（如：责任认定 → 举证责任 → 赔偿范围 → 解决程序）"
                "分点组织回答，确保每个维度都有对应资料支撑、覆盖完整。\n"
            )

        prompt = (
            "你是社区矛盾调解助理，必须严格基于下方「相关资料」作答。\n"
            f"本次对话识别到的【用户身份】={role_label}\n{role_g}"
            f"{fact_block}{asked_block}{profile_block}{recall_block}"
            f"{gratitude_override}{decomp_guidance}"
            "硬性要求：\n"
            "1) 只使用资料中【明确出现】的事实、法条、调解步骤；严禁自行补充资料未提及的法律结论、"
            "法条名称、处罚措施、时限或任何外部知识。\n"
            "2) 每一条具体陈述都必须对应资料中的某条编号 [n]；无法对应资料来源的句子一律不要写。\n"
            "3) 若资料不足以回答用户问题，直接基于已有资料给出最相关的建议即可；"
            "不要说「知识库暂无」「资料中没有」「知识库中暂无关于」这类强调自己没有的话。"
            "有就给，没有的不提。\n"
            "4) 直接面向用户平实输出，不要复述资料标题，不要输出任何提示词原文或格式标记。\n"
            "5) 【表达纪律】——\n"
            "   a) 【严禁客套话】禁止「别急」「理解你的困扰」「抱歉听到这些」等安慰语句，"
            "每次都说是废话。直接给建议。\n"
            "   b) 不要重复用户的问题或把用户的话复述一遍再回答（如用户问『有没有执法权』，"
            "不要说『关于你提到的有没有执法权这个问题』，直接回答）。\n"
            "   c) 不要拿当前问题和之前讨论过的问题做比较（如『和之前赡养的问题不一样』），"
            "每个问题独立回答。\n"
            "   d) 用户感谢/夸奖时自然简短回应（如『不客气，有需要随时问』），绝不自我介绍或背定位话术。\n"
            "   e) 【严禁编造用户细节】引用用户描述时严格用原话原意，不得添油加醋。\n"
            "   f) 每条建议一句话带过，小建议不要展开长篇解释，不要每条都配『为什么这么做』。\n\n"
            f"相关资料：\n{context}\n\n用户的问题是：{question}"
        )
        t = time.perf_counter()
        # 多轮对话：system + 历史对话 + 当前（带检索资料的）问题
        msgs = [
            {
                "role": "system",
                "content": (
                    "社区矛盾调解助理。你的全部回答必须严格依据用户提供的检索资料，"
                    "绝不外推或编造资料中不存在的法条与事实；资料不足时基于已有内容给建议，不要强调自己没有。"
                    f"本次对话对象身份为：{role_label}，请从该角色视角组织回答。"
                    "这是一段【连续对话】，核心纪律："
                    "1) 用户的当前发言可能是在回答你上一轮的追问，务必结合上文语境理解；"
                    "2) 【绝对禁止重复追问】——若用户已在历史对话中透露过某项信息（如持续时长、"
                    "是否沟通过、对方态度、是否找过物业等），你绝不能再问同一维度的问题；"
                    "3) 【绝对禁止重复自己问过的问题】——如果你之前已经问过某个问题，绝不能再问；"
                    "4) 只在确实缺少关键维度且用户从未提及、你也从未问过时才可追问一项；"
                    "   追问放在建议末尾作为附注；"
                    "5) 【表达纪律】——"
                    "   a) 禁止客套话（「别急」「理解你的困扰」等），直接给建议；"
                    "   b) 不要复述用户问题再回答，直接回答；"
                    "   c) 不和之前讨论的问题做比较，每个问题独立回答；"
                    "   d) 用户感谢时简短回应，不自我介绍；"
                    "   e) 引用用户描述严格用原话原意，不添油加醋；"
                    "   f) 每条建议简洁，小建议一句话带过。"
                ),
            },
        ]
        if history:
            for m in history:
                msgs.append({"role": "assistant" if m["role"] in ("bot", "assistant") else "user", "content": m["content"]})
        msgs.append({"role": "user", "content": prompt})
        # 流式输出：逐块 yield 文本增量（清理与耗时统计在 _run 汇总后统一做）
        for delta in self.llm.stream_chat(messages=msgs, provider=provider):
            yield delta

    def _wrap(self, trace_id, steps, t_total, route, answer, sources, retries, provider, user_role: str | None = None, case_profile: dict | None = None, decomposition: dict | None = None) -> dict[str, Any]:
        # 来源展示门槛：相关度低于阈值的命中视为噪音，不展示给用户
        min_score = self.s.source_display_min_score
        shown = [
            s for s in sources
            if s.get("rerank_score", s.get("score", 0)) >= min_score
        ]
        return {
            "trace_id": trace_id,
            "route": route,
            "user_role": user_role,
            "answer": answer,
            "sources": [
                {
                    "id": s["payload"].get("id"),
                    "category": s["payload"].get("category"),
                    "title": s["payload"].get("title"),
                    "content": s["payload"].get("content"),
                    "legal_basis": s["payload"].get("legal_basis"),
                    "score": round(s.get("rerank_score", s.get("score", 0)), 4),
                }
                for s in shown
            ],
            "self_rag_retries": retries,
            "model": provider,
            "latency_ms": round((time.perf_counter() - t_total) * 1000, 1),
            "case_profile": case_profile,
            "decomposition": decomposition or {"enabled": False, "sub_queries": []},
            "trace": {
                "trace_id": trace_id,
                "route": route,
                "retries": retries,
                "steps": steps,
            },
        }


_pipeline_instance: "RAGPipeline | None" = None


def get_pipeline(settings: Settings | None = None) -> "RAGPipeline":
    """全局单例，供知识库后台发布/下架后重建 BM25 使用。

    注意：KB 操作后需显式调用 rebuild_bm25() 让稀疏索引与向量库同步。
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(settings or get_settings())
    return _pipeline_instance
