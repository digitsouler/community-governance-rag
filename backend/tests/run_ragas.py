"""4.RAGAS 评测打分脚本（在本机运行）。

前置（本机一次性）：
  # 锁定 0.2.x：0.1.x import 阶段会因 langchain_community 新版路径崩溃；0.3.x 又改了 API
  pip install "ragas>=0.2,<0.3" langchain-openai
  # 评分用的 DEEPSEEK_API_KEY / ZHIPU_API_KEY 会自动从 backend/.env 读取，无需手动 export

RAGAS 评分需要 LLM + Embedding。默认示例用 OpenAI，但本项目面向国内模型，
下面给出用已有 key 的两种配置（二选一，改 main() 里的 llm / emb 两行即可）：

方案 A · 用 DeepSeek 当评分 LLM + 智谱 embedding-3 当评分 Embedding：
  from langchain_openai import ChatOpenAI, OpenAIEmbeddings
  llm = ChatOpenAI(model="deepseek-chat",
                   api_key=os.getenv("DEEPSEEK_API_KEY"),
                   base_url="https://api.deepseek.com/v1", temperature=0)
  emb = OpenAIEmbeddings(model="embedding-3",
                         api_key=os.getenv("ZHIPU_API_KEY"),
                         base_url="https://open.bigmodel.cn/api/paas/v4")

方案 B · 直接用 OpenAI（若你有 key）：
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  emb = OpenAIEmbeddings(model="text-embedding-3-small")

用法：
  cd backend
  PYTHONPATH=. python tests/run_ragas.py --provider all
  PYTHONPATH=. python tests/run_ragas.py --provider deepseek --out benchmark_report.md

评测维度（RAGAS 标准四指标）：
  - faithfulness        忠实度：答案是否严格基于检索上下文，不胡编
  - answer_relevancy    答案相关性：是否切题、回答了问题
  - context_precision   上下文精确度：检索到的相关文档排得靠前吗
  - context_recall      上下文召回率：标准答案中的信息是否都在检索上下文中

输入：eval_results/answers_{provider}.jsonl（由 gen_answers.py 生成）
      app/data/eval_set.json（取 reference_answer 作 ground_truth）
输出：benchmark_report.md（三模型 × 四指标横评表 + 逐条明细）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "eval_results"
EVAL_PATH = BASE / "app" / "data" / "eval_set.json"
PROVIDERS = ["deepseek", "zhipu", "qwen"]


def load_eval_map() -> dict[str, dict]:
    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return {c["id"]: c for c in ev["cases"]}


def load_answers(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        print(f"[.env] 未找到 {path}，将直接使用系统环境变量")
        return
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    print(f"[.env] 已从 {path.name} 载入 {n} 个变量")


def build_dataset(rows: list[dict], ev_map: dict) -> list[dict]:
    """转成 RAGAS 需要的样本：question/answer/contexts/ground_truth。"""
    out = []
    for r in rows:
        c = ev_map.get(r["id"], {})
        if r["expected_route"] != "retrieve":
            continue
        if not r["contexts"]:
            continue
        out.append({
            "question": r["query"],
            "answer": r["answer"],
            "contexts": [ctx["content"] for ctx in r["contexts"]],
            "ground_truth": c.get("reference_answer") or "",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="all", choices=PROVIDERS + ["all"])
    ap.add_argument("--out", default="benchmark_report.md")
    args = ap.parse_args()
    provs = PROVIDERS if args.provider == "all" else [args.provider]

    _load_dotenv(BASE / ".env") 

    # 延迟导入，确保本机装了 ragas 才 import
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    import os
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset 

    # 默认用 DeepSeek 当评分 LLM + 智谱 embedding-3 当评分 Embedding（国内模型可直接跑）
    # 若用 OpenAI，改回下方注释的两行：
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
    emb = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # emb = OpenAIEmbeddings(model="text-embedding-3-small")

    ev_map = load_eval_map()
    report_lines = ["# RAGAS 三模型横评报告", ""]
    report_lines.append("| 模型 | faithfulness | answer_relevancy | context_precision | context_recall |")
    report_lines.append("|---|---|---|---|---|")

    for p in provs:
        path = OUT_DIR / f"answers_{p}.jsonl"
        if not path.exists():
            print(f"[跳过] 缺少 {path.name}")
            continue
        rows = load_answers(path)
        data = build_dataset(rows, ev_map)
        if not data:
            print(f"[{p}] 无可用样本")
            continue
        print(f"[{p}] 评测样本 {len(data)} 条 …")
        # ragas >=0.2 的新 API：llm/embeddings 不再作为 evaluate() 的参数，
        # 而是挂到每个 metric 对象上；且输入必须是 HuggingFace Dataset。
        for _m in (faithfulness, answer_relevancy, context_precision, context_recall):
            _m.llm = llm
            _m.embeddings = emb
        result = evaluate(
            Dataset.from_list(data),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=emb,
        )
        df = result.to_pandas()
        means = {m: df[m].mean() for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
        report_lines.append(
            f"| {p} | {means['faithfulness']:.3f} | {means['answer_relevancy']:.3f} | "
            f"{means['context_precision']:.3f} | {means['context_recall']:.3f} |"
        )
        # 逐条明细
        report_lines.append("")
        report_lines.append(f"## {p} 逐条明细")
        report_lines.append("")
        report_lines.append("| id | faithfulness | answer_relevancy | context_precision | context_recall |")
        report_lines.append("|---|---|---|---|---|")
        for i, r in df.iterrows():
            report_lines.append(
                f"| {data[i]['question'][:24]} | {r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
                f"{r['context_precision']:.3f} | {r['context_recall']:.3f} |"
            )

    out_path = BASE / args.out
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已写入 {out_path}")


if __name__ == "__main__":
    main()
