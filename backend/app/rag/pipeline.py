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
    def _supervise(self, question: str) -> str:
        q = question.strip()
        ql = q.lower()
        greet = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "感谢"]
        if any(g in ql for g in greet) and len(q) <= 12:
            return "direct"
        if any(k in q for k in ["你是谁", "你是什么", "你能干", "你会", "介绍下你", "怎么用"]):
            return "direct"
        if len(q) < 4:
            return "clarify"
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

    # ---------- Retrieval + Self-RAG ----------
    def _reformulate(self, query: str) -> str:
        q = query
        for w in STOPWORDS:
            q = q.replace(w, "")
        return q.strip() or query

    # ---------- 对外接口 ----------
    def query(self, question: str, provider: ProviderName | None = None) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex[:12]
        provider = provider or self.s.default_llm
        steps: list[dict[str, Any]] = []
        t_total = time.perf_counter()

        def mark(stage: str, detail: str = "", start: float | None = None):
            ms = (time.perf_counter() - start) * 1000 if start else None
            steps.append({"stage": stage, "detail": detail, "ms": round(ms, 1) if ms is not None else None})

        log.info("[%s] 新请求 | provider=%s | q=%r", trace_id, provider, question[:60])

        # Supervisor 路由
        t = time.perf_counter()
        route = self._supervise(question)
        mark("supervise", f"route={route}", t)
        log.info("[%s] 路由判定=%s", trace_id, route)

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
        query = question
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

        answer = self._generate(question, ranked, provider, trace_id, steps)
        log.info("[%s] 完成 | 路由=retrieve 来源数=%d 重试=%d", trace_id, len(ranked), retries)
        return self._wrap(trace_id, steps, t_total, "retrieve", answer, ranked, retries, provider)

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

    def _generate(self, question: str, sources: list[dict], provider: str, trace_id: str, steps: list) -> str:
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
        prompt = (
            "你是社区矛盾调解助理，必须严格基于下方「相关资料」作答。\n"
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
        answer = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "社区矛盾调解助理。你的全部回答必须严格依据用户提供的检索资料，"
                        "绝不外推或编造资料中不存在的法条与事实；资料不足时如实告知。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            provider=provider,
        )
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

    def _wrap(self, trace_id, steps, t_total, route, answer, sources, retries, provider) -> dict[str, Any]:
        # 来源展示门槛：相关度低于阈值的命中视为噪音，不展示给用户
        min_score = self.s.source_display_min_score
        shown = [
            s for s in sources
            if s.get("rerank_score", s.get("score", 0)) >= min_score
        ]
        return {
            "trace_id": trace_id,
            "route": route,
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
