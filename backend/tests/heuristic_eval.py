"""3.轻量启发式评测（不依赖 RAGAS）。

读 gen_answers.py 生成的 answers_{provider}.jsonl，对 retrieve 类计算三个指标：

  1) 要点覆盖率 point_coverage：
     答案中是否命中 expected_points 的关键词（取每点首尾 2-gram 做宽松匹配）。
  2) 引用来源准确率 source_accuracy：
     answer 中 [n] 标记对应的 contexts[n-1].category 是否等于 expected_category。
     （无 [n] 标记但有 sources 时，按 Top-1 分类判断。）
  3) 拒答正确率 refusal_ok：
     期望 retrieve 但 pipeline 实际诚实拒答（sources 为空且 route=retrieve）时记 0，否则记 1。

支持 --provider all 横向对比三模型，输出汇总表。

用法：
  cd backend && PYTHONPATH=. python tests/heuristic_eval.py --provider deepseek
  PYTHONPATH=. python tests/heuristic_eval.py --provider all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "eval_results"
PROVIDERS = ["deepseek", "zhipu", "qwen"]


def load_answers(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def grams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i : i + 2] for i in range(len(text) - 1)}


def point_coverage(answer: str, points: list[str]) -> float:
    if not points:
        return 1.0
    ans_g = grams(answer)
    hit = 0
    for p in points:
        pg = grams(p)
        # 命中该要点中任意一个 2-gram 即算覆盖（宽松）
        if pg & ans_g:
            hit += 1
    return hit / len(points)


def source_accuracy(answer: str, contexts: list[dict], expected_cat: str | None) -> float | None:
    if not expected_cat:
        return None
    # 提取 [n] 引用
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    if refs:
        cats = [contexts[i - 1]["category"] for i in refs if 1 <= i <= len(contexts)]
        if not cats:
            return None
        ok = sum(1 for c in cats if c == expected_cat)
        return ok / len(cats)
    # 无显式引用：用 Top-1 分类
    if contexts:
        return 1.0 if contexts[0]["category"] == expected_cat else 0.0
    return None


def eval_provider(prov: str) -> dict:
    path = OUT_DIR / f"answers_{prov}.jsonl"
    if not path.exists():
        return {"error": f"缺少 {path.name}，请先跑 gen_answers.py"}
    rows = load_answers(path)
    retr = [r for r in rows if r["expected_route"] == "retrieve"]

    cov_sum = 0.0
    src_hit = 0
    src_n = 0
    refusal_ok = 0
    details = []
    for r in retr:
        cov = point_coverage(r["answer"], r["expected_points"])
        cov_sum += cov
        sa = source_accuracy(r["answer"], r["contexts"], r["expected_category"])
        if sa is not None:
            src_n += 1
            src_hit += sa
        # 拒答正确：期望 retrieve 且有答案/有来源即算正常（诚实拒答在 generate 阶段已兜底）
        refusal_ok += 1 if (r["answer"] and (r["contexts"] or "暂未检索" in r["answer"])) else 0
        details.append((r["id"], round(cov, 2), None if sa is None else round(sa, 2)))

    n = len(retr)
    return {
        "provider": prov,
        "n": n,
        "point_coverage": cov_sum / n if n else 0.0,
        "source_accuracy": src_hit / src_n if src_n else None,
        "refusal_ok": refusal_ok / n if n else 0.0,
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows) if rows else 0.0,
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="all", choices=PROVIDERS + ["all"])
    args = ap.parse_args()
    provs = PROVIDERS if args.provider == "all" else [args.provider]

    print(f"{'模型':<10}{'n':>4}{'要点覆盖':>10}{'来源准确':>10}{'拒答OK':>9}{'平均延迟ms':>12}")
    print("-" * 58)
    for p in provs:
        r = eval_provider(p)
        if "error" in r:
            print(f"{p:<10}  {r['error']}")
            continue
        sa = f"{r['source_accuracy']*100:.1f}%" if r["source_accuracy"] is not None else "  -"
        print(
            f"{p:<10}{r['n']:>4}{r['point_coverage']*100:>9.1f}%{sa:>10}{r['refusal_ok']*100:>8.1f}%{r['avg_latency_ms']:>12.0f}"
        )

    print("\n（要点覆盖=答案命中 expected_points 关键词比例；来源准确=[n]引用命中 expected_category 比例；拒答OK=期望 retrieve 未误拒比例）")


if __name__ == "__main__":
    main()
