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
from app.rag.llm import LLMClient
from app.rag.rerank import rerank
from app.rag.vectorstore import get_vector_store

log = get_logger("rag.pipeline")

STOPWORDS = ["我想问", "请问", "帮我", "怎么", "如何", "怎么办", "吗", "呢", "？", "?", "。", "居民", "社区", "小区"]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.embedder = EmbeddingClient(self.s)
        self.store = get_vector_store(self.s)
        self.llm = LLMClient(self.s)

    # ---------- Supervisor ----------
    def _supervise(self, question: str) -> str:
        q = question.strip()
        greet = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "感谢"]
        if any(g in q.lower() for g in greet) and len(q) <= 12:
            return "direct"
        if any(k in q for k in ["你是谁", "你是什么", "你能干", "你会", "介绍下你", "怎么用"]):
            return "direct"
        if len(q) < 4:
            return "clarify"
        return "retrieve"

    def _direct_answer(self) -> str:
        return (
            "我是社区矛盾调解助理，专注于邻里纠纷、物业矛盾、家庭赡养等"
            "社区治理场景的调解支持。你可以直接描述遇到的矛盾（例如"
            "「楼上漏水导致我家天花板发霉怎么办」），我会结合知识库给出"
            "处置建议、相关法条与调解步骤，并标注依据来源。"
        )

    def _clarify(self) -> str:
        return (
            "您的问题信息较少，难以精准检索。请补充：① 矛盾类型"
            "（噪音/漏水/停车/宠物/物业费/赡养等）；② 关键事实"
            "（谁、什么行为、造成什么影响）。例如：「一楼私装地锁占用公共车位，"
            "其他业主该如何处理？」"
        )

    # ---------- Retrieval + Self-RAG ----------
    def _retrieve(self, query: str, retry: int = 0) -> tuple[list[dict], float]:
        vec = self.embedder.embed_query(query)
        candidates = self.store.search(vec, top_k=self.s.top_k)
        ranked = rerank(query, candidates, top_k=self.s.rerank_top_k)
        best = ranked[0]["rerank_score"] if ranked else 0.0
        return ranked, best

    def _reformulate(self, query: str) -> str:
        q = query
        for w in STOPWORDS:
            q = q.replace(w, "")
        return q.strip() or query

    def _generate(self, question: str, sources: list[dict]) -> str:
        ctx_blocks = []
        for i, s in enumerate(sources, 1):
            p = s["payload"]
            steps = "；".join(p.get("mediation_steps", []))
            ctx_blocks.append(
                f"[{i}]（编号 {p.get('id')}｜{p.get('category')}）\n"
                f"标题：{p.get('title')}\n内容：{p.get('content')}\n"
                f"法条：{p.get('legal_basis')}\n步骤：{steps}"
            )
        context = "\n\n".join(ctx_blocks)
        prompt = (
            "你是社区矛盾调解助理。请严格依据下列【参考依据】回答用户问题，"
            "要求：1) 给出可操作的处置建议；2) 引用依据用 [1][2] 标注；"
            "3) 若依据不足，明确说明，不得编造法条或事实。\n\n"
            f"【参考依据】\n{context}\n\n用户问题：{question}"
        )
        return self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "社区矛盾调解助理，基于知识库提供有据可循的调解建议。",
                },
                {"role": "user", "content": prompt},
            ]
        )

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

        # retrieve + Self-RAG 重试
        # mock 模式下向量为字符哈希、无语义，阈值归零以便演示完整检索链路
        is_mock = self.embedder.use_mock
        thr = 0.0 if is_mock else self.s.relevance_threshold
        query = question
        ranked, best = self._retrieve(query, trace_id, steps)
        retries = 0
        while best < thr and retries < self.s.max_retrieve_retries:
            query = self._reformulate(query)
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
        t_emb = time.perf_counter()
        candidates = self.store.search(vec, top_k=self.s.top_k)
        ranked = rerank(query, candidates, top_k=self.s.rerank_top_k)
        best = ranked[0]["rerank_score"] if ranked else 0.0
        label = f"retry={retry} " if retry else ""
        log.info("[%s] 检索%s| 候选=%d 命中top=%d 最佳分=%.4f", trace_id, label, len(candidates), len(ranked), best)
        mark = "retrieve"
        if retry:
            mark = f"retrieve_r{retry}"
        steps.append({
            "stage": mark,
            "detail": f"候选={len(candidates)} top={len(ranked)} best={best:.4f}",
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
        prompt = (
            "你是社区矛盾调解助理。请严格依据下列【参考依据】回答用户问题，"
            "要求：1) 给出可操作的处置建议；2) 引用依据用 [1][2] 标注；"
            "3) 若依据不足，明确说明，不得编造法条或事实。\n\n"
            f"【参考依据】\n{context}\n\n用户问题：{question}"
        )
        t = time.perf_counter()
        answer = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "社区矛盾调解助理，基于知识库提供有据可循的调解建议。",
                },
                {"role": "user", "content": prompt},
            ],
            provider=provider,
        )
        steps.append({
            "stage": "generate",
            "detail": f"model={provider} 字数={len(answer)}",
            "ms": round((time.perf_counter() - t) * 1000, 1),
        })
        return answer

    def _wrap(self, trace_id, steps, t_total, route, answer, sources, retries, provider) -> dict[str, Any]:
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
                for s in sources
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
