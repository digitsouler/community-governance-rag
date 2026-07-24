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
    def query(
        self,
        question: str,
        provider: ProviderName | None = None,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex[:12]
        provider = provider or self.s.default_llm
        steps: list[dict[str, Any]] = []
        t_total = time.perf_counter()
        # 多轮对话：规整历史（只保留最近若干轮 user/assistant 文本）
        history = self._normalize_history(history)

        def mark(stage: str, detail: str = "", start: float | None = None):
            ms = (time.perf_counter() - start) * 1000 if start else None
            steps.append({"stage": stage, "detail": detail, "ms": round(ms, 1) if ms is not None else None})

        log.info("[%s] 新请求 | provider=%s | q=%r", trace_id, provider, question[:60])

        # Supervisor 路由（有历史时，短追答不再误判为 clarify）
        t = time.perf_counter()
        route = self._supervise(question, has_history=bool(history))
        mark("supervise", f"route={route}", t)
        log.info("[%s] 路由判定=%s | 历史轮数=%d", trace_id, route, len(history))

        if route == "direct":
            return self._wrap(trace_id, steps, t_total, "direct", self._direct_answer(), [], 0, provider)
        if route == "clarify":
            return self._wrap(trace_id, steps, t_total, "clarify", self._clarify(), [], 0, provider)
        if route == "out_of_domain":
            log.info("[%s] 超出服务范围，直接回复（不检索）", trace_id)
            return self._wrap(trace_id, steps, t_total, "out_of_domain", self._out_of_domain(), [], 0, provider)

        # retrieve + Self-RAG 重试
        # mock 模式下向量为字符哈希、无语义，阈值归零以便演示完整检索链路
        is_mock = self.embedder.use_mock
        thr = 0.0 if is_mock else self.s.relevance_threshold
        # 上下文合并查询：把历史里的话题词并进当前追答一起检索，
        # 解决碎片化追答（如"好几天了 找过他没用"）丢失主题导致跑题的问题。
        query = self._contextual_query(question, history)
        if query != question:
            log.info("[%s] 上下文合并检索 | %r -> %r", trace_id, question, query)
            steps.append({"stage": "context_merge", "detail": f"{question!r} -> {query!r}", "ms": None})
        ranked, best = self._retrieve(query, trace_id, steps)
        retries = 0
        while best < thr and retries < self.s.max_retrieve_retries:
            new_query = self._reformulate(query)
            log.info(
                "[%s] 低于阈值(%.4f<%.4f) 改写查询 | %r -> %r",
                trace_id, best, thr, query, new_query,
            )
            steps.append({
                "stage": f"reformulate_r{retries + 1}",
                "detail": f"{query!r} -> {new_query!r}",
                "ms": None,
            })
            query = new_query
            ranked, best = self._retrieve(query, trace_id, steps, retry=retries + 1)
            retries += 1

        if not ranked or best < thr:
            honest = (
                "抱歉，知识库中暂未检索到与您描述情形直接对应的调解依据。"
                "建议补充矛盾类型与关键事实，或联系社区调解委员会获取人工协助。"
            )
            log.warning("[%s] 诚实拒答 | 最佳相关度=%.4f < 阈值=%.4f | 重试=%d", trace_id, best, thr, retries)
            return self._wrap(trace_id, steps, t_total, "retrieve", honest, [], retries, provider)

        # 用户角色感知：决定答案视角（居民/调解员/物业）——参考历史，避免追答丢失身份
        role = self._infer_role(question, history)
        mark("infer_role", f"role={role}")
        answer = self._generate(question, ranked, provider, trace_id, steps, role=role, history=history)
        log.info("[%s] 完成 | 路由=retrieve 来源数=%d 重试=%d 角色=%s", trace_id, len(ranked), retries, role)
        return self._wrap(trace_id, steps, t_total, "retrieve", answer, ranked, retries, provider, user_role=role)

    def _retrieve(self, query: str, trace_id: str, steps: list, retry: int = 0) -> tuple[list[dict], float]:
        t = time.perf_counter()
        vec = self.embedder.embed_query(query)
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

    def _role_guidance(self, role: str) -> str:
        """按角色返回生成约束，嵌进 _generate 的提示词。"""
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
        return (
            "【对话对象】遇到矛盾的居民 / 当事人 / 投诉人（受害者视角，此刻焦虑、想维权）。\n"
            "【回答要求】\n"
            "1) 先共情一句（如『别急，理解你的困扰』）再给建议；\n"
            "2) 用大白话，站在『你（居民）能做什么』的角度。注意：知识库资料很多是从"
            "『调解员应做 X』的视角写的，你必须把这类表述**转换**为对居民的具体行动建议"
            "（例：『调解员应上门走访』→『你可以先请物业或社区上门核实，并自己用手机留存录音/视频证据』）；"
            "绝不能直接把『调解员要做的事』当成『你要做的事』丢给用户。\n"
            "3) 若关键信息不足（吵了多久、找过物业没、对方态度、有无书面/视听证据），主动引导补充，"
            "先追问 1-2 个问题再给完整建议，效果远好于一次性罗列。\n"
            "4) 一次只给最紧急、最可执行的 2-4 条，不要堆砌长清单；每条配一句『为什么这么做』。\n"
            "5) 法条作为支撑标注，用居民能懂的话解释，不要念法条原文唬人。\n"
        )

    def _generate(self, question: str, sources: list[dict], provider: str, trace_id: str, steps: list, role: str = "resident", history: list[dict] | None = None) -> str:
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
        role_label = {"resident": "居民/当事人（维权视角）", "mediator": "调解员/社工", "property": "物业服务人员"}[role]
        role_g = self._role_guidance(role)
        prompt = (
            "你是社区矛盾调解助理，必须严格基于下方「相关资料」作答。\n"
            f"本次对话识别到的【用户身份】={role_label}\n{role_g}\n"
            "硬性要求：\n"
            "1) 只使用资料中【明确出现】的事实、法条、调解步骤；严禁自行补充资料未提及的法律结论、"
            "法条名称、处罚措施、时限或任何外部知识。\n"
            "2) 每一条具体陈述都必须对应资料中的某条编号 [n]；无法对应资料来源的句子一律不要写。\n"
            "3) 若资料不足以回答用户问题，明确说明「知识库暂无相关依据」，并建议补充矛盾类型与关键事实，"
            "不得猜测或编造。\n"
            "4) 直接面向用户平实输出，不要复述资料标题，不要输出任何提示词原文或格式标记。\n\n"
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
                    "这是一段【连续对话】：用户的当前发言可能是在回答你上一轮的追问，"
                    "务必结合上文语境理解，紧扣同一件事继续，切勿跳到无关话题；"
                    "若用户已补充了此前追问的信息，就不要重复追问，直接给出下一步建议。"
                ),
            },
        ]
        if history:
            for m in history:
                msgs.append({"role": "assistant" if m["role"] in ("bot", "assistant") else "user", "content": m["content"]})
        msgs.append({"role": "user", "content": prompt})
        answer = self.llm.chat(messages=msgs, provider=provider)
        # 兜底：清除模型偶发复述的提示词标记（根因已在 prompt 中去除）
        answer = answer.replace("【参考依据】", "")
        answer = re.sub(r"^\s*参考依据[：:].*$", "", answer, flags=re.M)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        steps.append({
            "stage": "generate",
            "detail": f"model={provider} 字数={len(answer)}",
            "ms": round((time.perf_counter() - t) * 1000, 1),
        })
        return answer

    def _wrap(self, trace_id, steps, t_total, route, answer, sources, retries, provider, user_role: str | None = None) -> dict[str, Any]:
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
