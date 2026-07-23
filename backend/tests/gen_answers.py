"""2.对评测集跑完整的 RAG 管道，输出供 RAGAS / 启发式评测用的 answers。

用法：
  cd backend && PYTHONPATH=. python tests/gen_answers.py --provider deepseek
  PYTHONPATH=. python tests/gen_answers.py --provider all          # 三模型全跑
  PYTHONPATH=. python tests/gen_answers.py --provider zhipu --only-retrieve   # 只跑 retrieve 类
  PYTHONPATH=. python tests/gen_answers.py --provider qwen --resume # 断点续跑（跳过已生成 id）

输出：
  backend/eval_results/answers_{provider}.jsonl
  每行一个 JSON：{id, query, expected_route, expected_category, expected_points,
                  actual_route, answer, contexts, latency_ms, retries, provider}

说明：
  - direct / clarify / out_of_domain 类不调用 LLM 生成，默认跳过（用 --include-non-retrieve 强制跑）。
  - contexts 字段保留 RAGAS 需要的检索上下文（id / category / title / content / score）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import MODEL_REGISTRY, ProviderName, get_settings
from app.data.ingest import ingest
from app.log import setup_logging
from app.rag.pipeline import RAGPipeline

BASE = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE / "app" / "data" / "eval_set.json"
OUT_DIR = BASE / "eval_results"

PROVIDERS = ["deepseek", "zhipu", "qwen"]


def load_cases() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["cases"]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=PROVIDERS + ["all"])
    ap.add_argument("--only-retrieve", action="store_true", help="只跑 retrieve 类（默认行为）")
    ap.add_argument("--include-non-retrieve", action="store_true", help="也跑 direct/clarify/out_of_domain")
    ap.add_argument("--resume", action="store_true", help="跳过已生成 id")
    args = ap.parse_args()

    setup_logging("warning")
    providers = PROVIDERS if args.provider == "all" else [args.provider]

    cases = load_cases()
    if not args.include_non_retrieve:
        cases = [c for c in cases if c["expected_route"] == "retrieve"]

    for prov in providers:
        OUT_DIR.mkdir(exist_ok=True)
        out_path = OUT_DIR / f"answers_{prov}.jsonl"
        done = load_done(out_path) if args.resume else set()
        if done:
            print(f"[{prov}] 续跑：已存在 {len(done)} 条，跳过")

        s = get_settings()
        # 该 provider 的 LLM key 缺失时跳过，避免空跑
        key_attr = MODEL_REGISTRY[prov]["key_env"]
        if not getattr(s, key_attr, ""):
            print(f"[{prov}] 跳过：未配置 {key_attr}")
            continue

        ingest(force=True)
        pipe = RAGPipeline(s)
        pname = prov  # Literal 运行时即 str，pipeline.query 直接收字符串

        n_ok = 0
        with out_path.open("a", encoding="utf-8") as f:
            for c in cases:
                cid = c["id"]
                if cid in done:
                    continue
                q = c["query"]
                try:
                    res = pipe.query(q, provider=pname)
                except Exception as e:  # 单条失败不影响整体
                    print(f"  ! {cid} 失败：{e}")
                    continue

                rec = {
                    "id": cid,
                    "query": q,
                    "expected_route": c["expected_route"],
                    "expected_category": c.get("expected_category"),
                    "expected_points": c.get("expected_points", []),
                    "reference_answer": c.get("reference_answer"),
                    "actual_route": res["route"],
                    "answer": res["answer"],
                    "contexts": [
                        {
                            "id": src["id"],
                            "category": src["category"],
                            "title": src["title"],
                            "content": src["content"],
                            "score": src["score"],
                        }
                        for src in res["sources"]
                    ],
                    "latency_ms": res["latency_ms"],
                    "retries": res["self_rag_retries"],
                    "provider": prov,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                print(f"  [{prov}] {cid} route={res['route']} ctx={len(res['sources'])} {res['latency_ms']:.0f}ms")

        print(f"[{prov}] 完成：新增 {n_ok} 条 -> {out_path}\n")


if __name__ == "__main__":
    main()
