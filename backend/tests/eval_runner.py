"""评测集离线跑分：验证路由准确率 + 检索命中质量（不调用 LLM 生成，省 token）。

用法：
  cd backend && PYTHONPATH=. python tests/eval_runner.py                 # 默认（混合检索，enable_hybrid）
  cd backend && PYTHONPATH=. python tests/eval_runner.py --mode vector   # 纯向量召回（旧版对照）
  cd backend && PYTHONPATH=. python tests/eval_runner.py --compare        # 两种模式对照，量化混合检索提升
可选：RELEVANCE_THRESHOLD=0.30 调整阈值观察拒答边界。

评测三件事：
  1) 路由准确率：_supervise(query) 是否等于 expected_route（与检索模式无关）
  2) 检索命中：retrieve 类中，Top-1 命中的语料分类是否等于 expected_category；
     并统计 best rerank 分数分布（衡量扩语料后分数是否更有区分度）。
  3) 混合检索对照：--compare 同时跑 vector / hybrid，对比类目命中率与 best 分。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.data.ingest import ingest
from app.log import setup_logging
from app.rag.pipeline import RAGPipeline

EVAL_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "eval_set.json"


def load_cases() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["cases"]


def run_mode(s, mode: str, cases: list[dict]) -> dict:
    """跑单个检索模式，返回统计。mode='vector'|'hybrid'。"""
    s.enable_hybrid = (mode == "hybrid")
    pipe = RAGPipeline(s)
    thr = s.relevance_threshold

    route_ok = 0
    retr_cases = 0
    cat_hit = 0
    scores: list[float] = []
    fails: list[str] = []

    print(f"\n===== 模式：{mode.upper()}（enable_hybrid={s.enable_hybrid}）=====")
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
            ranked, best = pipe._retrieve(q, "eval", [])
            if ranked:
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
    route_acc = route_ok / total * 100
    cat_acc = cat_hit / retr_cases * 100 if retr_cases else 0.0
    print(f"\n路由准确率：{route_ok}/{total} = {route_acc:.1f}%")
    if retr_cases:
        print(f"检索类目命中率(Top-1)：{cat_hit}/{retr_cases} = {cat_acc:.1f}%")
    if scores:
        scores_sorted = sorted(scores)
        mean = sum(scores) / len(scores)
        print(f"\nbest 分数分布（retrieve 类，阈值={thr}）：")
        print(f"  最低={min(scores):.4f}  最高={max(scores):.4f}  中位={scores_sorted[len(scores)//2]:.4f}  均值={mean:.4f}")
        below = [x for x in scores if x < thr]
        print(f"  低于阈值 {thr} 的条目数：{len(below)}（这些会触发 Self-RAG 改写重试）")
    if fails:
        print("\n未通过项：")
        for f in fails:
            print("  -", f)
    else:
        print("\n全部通过 ✅")

    return {
        "mode": mode,
        "route_ok": route_ok,
        "total": total,
        "route_acc": route_acc,
        "retr_cases": retr_cases,
        "cat_hit": cat_hit,
        "cat_acc": cat_acc,
        "scores": scores,
        "fails": fails,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="矛盾调解 RAG 离线评测")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default=None,
                        help="检索模式；缺省则使用配置中的 enable_hybrid")
    parser.add_argument("--compare", action="store_true", help="同时跑 vector 与 hybrid 对照")
    args = parser.parse_args()

    setup_logging("warning")  # 评测批量跑，压低日志噪音，只看结果表
    s = get_settings()
    if args.mode:
        s.enable_hybrid = (args.mode == "hybrid")

    n = ingest(force=True)
    print(f"已入库语料 {n} 条")

    cases = load_cases()

    if args.compare:
        r_vec = run_mode(s, "vector", cases)
        r_hyb = run_mode(s, "hybrid", cases)
        print("\n" + "=" * 60)
        print("对照结论（混合检索 vs 纯向量）")
        print("=" * 60)
        print(f"路由准确率      ：vector {r_vec['route_acc']:.1f}%  | hybrid {r_hyb['route_acc']:.1f}%  （路由与检索模式无关）")
        print(f"类目命中率(Top-1)：vector {r_vec['cat_acc']:.1f}%  | hybrid {r_hyb['cat_acc']:.1f}%")
        if r_vec["scores"] and r_hyb["scores"]:
            mv = sum(r_vec["scores"]) / len(r_vec["scores"])
            mh = sum(r_hyb["scores"]) / len(r_hyb["scores"])
            print(f"best 均值        ：vector {mv:.4f}  | hybrid {mh:.4f}  （Δ={mh - mv:+.4f}）")
        # 找出 hybrid 纠错 / 退化的条目
        vec_fail = {f.split(":")[0] for f in r_vec["fails"]}
        hyb_fail = {f.split(":")[0] for f in r_hyb["fails"]}
        improved = vec_fail - hyb_fail
        regressed = hyb_fail - vec_fail
        if improved:
            print(f"混合检索纠错（vector 错→hybrid 对）：{sorted(improved)}")
        if regressed:
            print(f"混合检索退化（vector 对→hybrid 错）：{sorted(regressed)} ⚠️")
        if not improved and not regressed:
            print("两类目命中一致，混合检索未改变 Top-1 结果（召回池扩大但精排稳定）")
    else:
        mode = "hybrid" if s.enable_hybrid else "vector"
        run_mode(s, mode, cases)


if __name__ == "__main__":
    main()
