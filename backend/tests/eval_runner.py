"""评测集离线跑分：验证路由准确率 + 检索命中质量（不调用 LLM 生成，省 token）。

用法：
  cd backend && PYTHONPATH=. python tests/eval_runner.py
可选：RELEVANCE_THRESHOLD=0.30 调整阈值观察拒答边界。

评测两件事：
  1) 路由准确率：_supervise(query) 是否等于 expected_route
  2) 检索命中：retrieve 类中，Top-1 命中的语料分类是否等于 expected_category；
     并统计 best rerank 分数分布（衡量扩语料后分数是否更有区分度）。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.data.ingest import ingest
from app.log import setup_logging
from app.rag.embeddings import EmbeddingClient
from app.rag.pipeline import RAGPipeline
from app.rag.rerank import rerank
from app.rag.vectorstore import get_vector_store

EVAL_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "eval_set.json"


def load_cases() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["cases"]


def main() -> None:
    setup_logging("warning")  # 评测批量跑，压低日志噪音，只看结果表
    s = get_settings()
    n = ingest(force=True)
    print(f"已入库语料 {n} 条\n")

    pipe = RAGPipeline(s)
    embedder = EmbeddingClient(s)
    store = get_vector_store(s)
    thr = s.relevance_threshold

    cases = load_cases()
    route_ok = 0
    retr_cases = 0
    cat_hit = 0
    scores: list[float] = []
    fails: list[str] = []

    print(f"{'ID':<22}{'期望路由':<14}{'实判路由':<14}{'命中分类':<18}{'best':>7}  判定")
    print("-" * 92)
    for c in cases:
        q = c["query"]
        exp_route = c["expected_route"]
        got_route = pipe._supervise(q)
        route_match = got_route == exp_route
        if route_match:
            route_ok += 1

        top_cat = "-"
        best = 0.0
        cat_match_str = ""
        if exp_route == "retrieve":
            retr_cases += 1
            vec = embedder.embed_query(q)
            cands = store.search(vec, top_k=s.top_k)
            ranked = rerank(q, cands, top_k=s.rerank_top_k)
            if ranked:
                best = ranked[0]["rerank_score"]
                top_cat = ranked[0]["payload"].get("category", "-")
                scores.append(best)
            exp_cat = c.get("expected_category")
            if top_cat == exp_cat:
                cat_hit += 1
                cat_match_str = "✓类目"
            else:
                cat_match_str = f"✗(期望{exp_cat})"

        verdict = "OK" if route_match else "✗路由"
        if exp_route == "retrieve":
            verdict = f"{'OK' if route_match else '✗路由'} {cat_match_str}"
        if not route_match or (exp_route == "retrieve" and top_cat != c.get("expected_category")):
            fails.append(f"{c['id']}: route {exp_route}->{got_route}, cat {c.get('expected_category')}->{top_cat}")

        print(f"{c['id']:<22}{exp_route:<14}{got_route:<14}{top_cat:<18}{best:>7.4f}  {verdict}")

    print("-" * 92)
    total = len(cases)
    print(f"\n路由准确率：{route_ok}/{total} = {route_ok/total*100:.1f}%")
    if retr_cases:
        print(f"检索类目命中率(Top-1)：{cat_hit}/{retr_cases} = {cat_hit/retr_cases*100:.1f}%")
    if scores:
        scores_sorted = sorted(scores)
        print(f"\nbest 分数分布（retrieve 类，阈值={thr}）：")
        print(f"  最低={min(scores):.4f}  最高={max(scores):.4f}  中位={scores_sorted[len(scores)//2]:.4f}  均值={sum(scores)/len(scores):.4f}")
        below = [x for x in scores if x < thr]
        print(f"  低于阈值 {thr} 的条目数：{len(below)}（这些会触发 Self-RAG 改写重试）")
    if fails:
        print("\n未通过项：")
        for f in fails:
            print("  -", f)
    else:
        print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
