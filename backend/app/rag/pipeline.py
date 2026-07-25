"""核心 RAG 管道（Agentic）。

两大差异化能力：
  1. Supervisor 路由：先判断问题类型，决定「直接回答 / 要求澄清 / 走检索」，
     避免所有问题都无脑检索（省 token、降幻觉）。
  2. Self-RAG 自纠错：检索结果不达标时自动改写查询重试；重试后仍不足则
     诚实告知「知识库暂无依据」，绝不编造。

对外暴露 query()：输入问题 + 模型供应商，输出答案、引用来源、路由决策与重试次数。
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any

from app.config import ProviderName, Settings, get_settings
from app.log import get_logger
from app.profile_store import get_profile_store
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

    # ---------- 对外接口 ----------
    def query(self, question, provider=None, history=None, session_id=None, stream=False):
        """统一入口。

        stream=False → 返回完整结果 dict（兼容旧调用 / 评测脚本）；
        stream=True  → 返回生成器，逐条 yield SSE 事件
                      {"type": "route"|"delta"|"done"|"error", ...}。
        """
        gen = self._run(question, provider, history, session_id)
        if stream:
            return gen
        final = None
        for ev in gen:
            if ev.get("type") == "done":
                final = ev["result"]
        return final

    def _run(self, question, provider, history, session_id):
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

        # retrieve + Self-RAG 重试
        is_mock = self.embedder.use_mock
        thr = 0.0 if is_mock else self.s.relevance_threshold
        query = self._contextual_query(question, history)
        if query != question:
            log.info("[%s] 上下文合并检索 | %r -> %r", trace_id, question, query)
            steps.append({"stage": "context_merge", "detail": f"{question!r} -> {query!r}", "ms": None})
        qvec = self.embedder.embed_query(query)
        ranked, best = self._retrieve(query, trace_id, steps, query_vec=qvec)
        retries = 0
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

        role = self._infer_role(question, history)
        mark("infer_role", f"role={role}")
        case_profile = self._profiles.get(session_id) if session_id else None

        # 检索链路已就绪，先把「路由 + 已完成步骤」推给前端（首字前可见骨架）
        yield {"type": "route", "route": "retrieve", "trace_id": trace_id, "session_id": session_id,
               "steps": list(steps)}

        # 流式生成答案（首字即可见，体感远快于等整段）
        answer_parts: list[str] = []
        t_gen = time.perf_counter()
        for delta in self._generate_stream(question, ranked, provider, trace_id, role=role, history=history, profile=case_profile):
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
        if session_id:
            existing = case_profile or {}
            fresh = self._build_case_profile(history, question, role)
            merged = self._merge_profile(existing, fresh)
            self._profiles.save(session_id, merged)
            case_profile = merged

        log.info("[%s] 完成 | 路由=retrieve 来源数=%d 重试=%d 角色=%s", trace_id, len(ranked), retries, role)
        wrapped = self._wrap(trace_id, steps, t_total, "retrieve", answer, ranked, retries, provider,
                             user_role=role, case_profile=case_profile)
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

    # ---------- 用户角色感知（Agent 能力：让答案服从对话对象身份） ----------
    def _infer_role(self, question: str, history: list[dict] | None = None) -> str:
        """从用户措辞推断其身份，用于决定答案视角。

        默认 'resident'（居民/当事人/投诉人）——因为我们面对的主要是来维权的居民；
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
        # 已知维度列表，用于在追问指令中排除
        known_dims = "、".join(known.keys()) if known else ""
        if role == "mediator":
            return (
                "【对话对象】社区调解员 / 社工。TA 需要接案处置流程与约谈技巧。\n"
                "【回答要求】按调解工作专业流程组织：受理登记要点 → 核实与走访 → 组织调解/座谈"
                " → 签订约定与回访。可直接引用知识库处置步骤原文，使用专业口吻，不必过度共情。\n"
            )
        if role == "property":
            return (
                "【对话对象】物业服务人员。TA 需要物业视角的处置动作（巡查、记录、上报、协助）。\n"
                "【回答要求】从物业职责角度给可执行动作：现场核实、台账记录、协调工程/安保、"
                "向业主反馈、上报社区等。使用物业工作口吻。\n"
            )
        # resident（默认）：居民 / 当事人 / 投诉人（维权视角）
        # 核心原则：【先给建议，追问是附注】。用户来这里是求解决方案的，
        # 不是来接受面试的。每轮都追问会让用户非常烦躁。
        ask_rule = (
            f"3) 【默认给建议】基于已有信息直接给出 2-4 条可执行建议。"
            f"只在信息确实严重不足、且不补充就给不出有效建议时，才可在建议末尾简短提一句还需要什么信息。"
            f"【严禁追问以下用户已告知的维度】：{known_dims}。"
            f"绝对不要每轮都追问，不要当面试官。"
            if known_dims else
            "3) 【默认给建议】基于已有信息直接给出 2-4 条可执行建议。"
            "不要每轮都追问用户问题——用户来这里是求解决方案的，不是来接受面试的。"
            "只在信息确实严重不足时，才可在建议末尾简短提一句还需要什么信息。"
        )
        return (
            "【对话对象】遇到矛盾的居民 / 当事人 / 投诉人（受害者视角，此刻焦虑、想维权）。\n"
            "【回答要求】\n"
            "1) 先共情一句（如『别急，理解你的困扰』）再给建议；\n"
            "2) 用大白话，站在『你（居民）能做什么』的角度。注意：知识库资料很多是从"
            "『调解员应做 X』的视角写的，你必须把这类表述**转换**为对居民的具体行动建议"
            "（例：『调解员应上门走访』→『你可以先请物业或社区上门核实，并自己用手机留存录音/视频证据』）；"
            "绝不能直接把『调解员要做的事』当成『你要做的事』丢给用户。\n"
            f"{ask_rule}\n"
            "4) 一次只给最紧急、最可执行的 2-4 条，不要堆砌长清单；每条配一句『为什么这么做』。\n"
            "5) 法条作为支撑标注，用居民能懂的话解释，不要念法条原文唬人。\n"
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

    def _generate_stream(self, question: str, sources: list[dict], provider: str, trace_id: str, role: str = "resident", history: list[dict] | None = None, profile: dict | None = None):
        """流式生成：逐块 yield 文本增量（mock 模式下整段一次性 yield）。

        纯感谢短路已上移到 _run，入口保证不会传入纯客套消息。
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
        role_label = {"resident": "居民/当事人（维权视角）", "mediator": "调解员/社工", "property": "物业服务人员"}[role]
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

        prompt = (
            "你是社区矛盾调解助理，必须严格基于下方「相关资料」作答。\n"
            f"本次对话识别到的【用户身份】={role_label}\n{role_g}"
            f"{fact_block}{asked_block}{profile_block}"
            f"{gratitude_override}"  # v6：感谢时强制覆盖
            "硬性要求：\n"
            "1) 只使用资料中【明确出现】的事实、法条、调解步骤；严禁自行补充资料未提及的法律结论、"
            "法条名称、处罚措施、时限或任何外部知识。\n"
            "2) 每一条具体陈述都必须对应资料中的某条编号 [n]；无法对应资料来源的句子一律不要写。\n"
            "3) 若资料不足以回答用户问题，明确说明「知识库暂无相关依据」，并建议补充矛盾类型与关键事实，"
            "不得猜测或编造。\n"
            "4) 直接面向用户平实输出，不要复述资料标题，不要输出任何提示词原文或格式标记。\n"
            "5) 【表达习惯】——\n"
            "   a) 安慰性语句（如\"别急""理解你的困扰""抱歉听到这些\"）只在对话首次出现，"
            "后续轮次直接给建议或推进下一步，不要再重复说。用户来是解决问题的，不是来听客套话的。\n"
            "   b) 当你发现某个关键信息在对话中从未被用户提及时（如是否沟通过、持续多久），"
            "应表述为「你在对话中还没有提到过…」，而不是「目前资料里没有提到…」——"
            "这是用户没说过，不是资料里没有。\n"
            "   c) 当用户表达感谢、满意或正面评价时（如\"谢谢""真好用""有帮助\"），"
            "自然简短回应即可（如\"不客气，有需要随时问」「能帮到你就好」），"
            "绝对不要趁机做自我介绍、背产品说明书或重复自己的定位，这会显得非常生硬和虚伪。\n"
            "   d) 【严禁编造用户未提及的细节】——用户在对话中描述的具体情况"
            "（时间、频次、影响、对方行为等），你在回答中引用时必须严格基于用户原话，"
            "不得添油加醋、夸大程度或补充用户没说过的形容词。"
            "例如用户说「看电视嗡嗡响」，不能写成「电视声音巨大」；"
            "用户说「9点搞到凌晨」，不能写成「每晚制造极大噪音」。\n\n"
            f"相关资料：\n{context}\n\n用户的问题是：{question}"
        )
        t = time.perf_counter()
        # 多轮对话：system + 历史对话 + 当前（带检索资料的）问题
        msgs = [
            {
                "role": "system",
                "content": (
                    "社区矛盾调解助理。你的全部回答必须严格依据用户提供的检索资料，"
                    "绝不外推或编造资料中不存在的法条与事实；资料不足时如实告知。"
                    f"本次对话对象身份为：{role_label}，请从该角色视角组织回答。"
                    "这是一段【连续对话】，核心纪律："
                    "1) 用户的当前发言可能是在回答你上一轮的追问，务必结合上文语境理解；"
                    "2) 【绝对禁止重复追问】——若用户已在历史对话中透露过某项信息（如持续时长、"
                    "是否沟通过、对方态度、是否找过物业等），你绝不能再问同一维度的问题，"
                    "这会让用户非常愤怒；直接基于已知信息推进到下一步建议；"
                    "3) 【绝对禁止重复自己问过的问题】——如果你在之前的回答中已经问过"
                    "「有没有录音/录像」「持续多久」「有没有找过物业」等问题，绝不能再问"
                    "同样或类似的问题，即使用户还没有回答。这会让你显得没有在听、非常不专业；"
                    "4) 只在确实缺少某个关键维度、且用户从未提及、你也从未问过时，才可追问一项；"
                    "   且追问必须放在建议的末尾作为附注，而不是把追问当成回答的主体；"
                    "5) 【表达纪律】——"
                    "   a) 安慰语（\"别急""理解你的困扰\"等）只在首次出现，后续直接进入正题；"
                    "   b) 区分「资料里没有」和「用户没提过」：前者说知识库暂无依据，后者说你在对话中未提及；"
                    "   c) 用户感谢/夸奖时自然简短回应，绝不趁机自我介绍或背定位话术；"
                    "   d) 【严禁编造用户细节】——引用用户描述的情况时严格用原话原意，"
                    "不得添油加醋、夸大程度或补充用户没说过的形容词。这会让用户觉得你在瞎编。"
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

    def _wrap(self, trace_id, steps, t_total, route, answer, sources, retries, provider, user_role: str | None = None, case_profile: dict | None = None) -> dict[str, Any]:
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
