"""1.为 eval_set.json 的 retrieve 类条目补 reference_answer。

策略：按 expected_category 聚合语料文档，取该分类下所有文档的 content + legal_basis
精简拼成一段标准化参考答案（保留关键处置要点与法条），落到对应 case 的 reference_answer。
对抗评测（纯法条条号）同样用其 expected_category 的语料生成，保证参考答案真实可答。

仅当 case 没有 reference_answer 时才写入，幂等可重跑。
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE / "app" / "data" / "eval_set.json"
CORPUS_PATH = BASE / "app" / "data" / "mediation_cases.json"


def main() -> None:
    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    # 按分类聚合语料文本
    by_cat: dict[str, list[dict]] = {}
    for d in corpus:
        by_cat.setdefault(d["category"], []).append(d)

    # 精简：每篇取 content 前 220 字 + legal_basis
    def build_ref(cat: str) -> str:
        docs = by_cat.get(cat, [])
        parts = []
        for d in docs:
            c = d.get("content", "").strip()
            c = c[:220] + ("…" if len(d.get("content", "")) > 220 else "")
            lb = d.get("legal_basis", "").strip()
            block = f"【{d.get('title','')}】{c}"
            if lb:
                block += f"\n依据：{lb}"
            parts.append(block)
        return "\n\n".join(parts)

    n_add = 0
    for case in ev["cases"]:
        if case["expected_route"] != "retrieve":
            continue
        if case.get("reference_answer"):
            continue
        cat = case["expected_category"]
        ref = build_ref(cat)
        case["reference_answer"] = ref
        n_add += 1

    EVAL_PATH.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已为 {n_add} 条 retrieve 类补 reference_answer；总计 retrieve 类={sum(1 for c in ev['cases'] if c['expected_route']=='retrieve')}")


if __name__ == "__main__":
    main()
