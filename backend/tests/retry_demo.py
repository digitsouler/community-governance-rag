"""演示 Self-RAG「低于阈值 → 改写重试」的完整日志链路。

两种结局都会演示到：
  ① 自纠错成功：首检 < 阈值 → 改写 → 复检越过阈值 → 生成答案（诚实说明无相关依据）
  ② 重试耗尽：阈值调严后，改写后仍 < 阈值 → 达到 max_retrieve_retries → 诚实拒答

用法（沙箱）：
  # 默认阈值 0.15（演示自纠错成功）
  cd backend && PYTHONPATH=. python tests/retry_demo.py
  # 调严阈值 0.30（演示重试耗尽 → 诚实拒答），可用环境变量覆盖：
  cd backend && RELEVANCE_THRESHOLD=0.30 PYTHONPATH=. python tests/retry_demo.py
"""
from __future__ import annotations

import os

# 必须在 import app 之前设置，pydantic-settings 才会读取到
THR = os.environ.get("RELEVANCE_THRESHOLD", "")
if THR:
    os.environ["RELEVANCE_THRESHOLD"] = THR

import logging  # noqa: E402

from app.data.ingest import ingest  # noqa: E402
from app.log import setup_logging  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402

setup_logging("debug")

print(f"\n>>> 当前 relevance_threshold={os.environ.get('RELEVANCE_THRESHOLD', '(默认 0.15)')}；先入库 16 条语料")
ingest(force=True)


def run(label: str, question: str, provider: str = "deepseek"):
    print("\n" + "=" * 72)
    print(f"【{label}】 q={question!r}")
    print("=" * 72)
    pipe = RAGPipeline()
    res = pipe.query(question, provider=provider)
    print("\n----- 返回摘要 -----")
    print("route            :", res["route"])
    print("self_rag_retries :", res["self_rag_retries"])
    print("sources          :", len(res["sources"]))
    print("answer           :", res["answer"][:140] + ("..." if len(res["answer"]) > 140 else ""))
    print("trace.steps      :")
    for s in res["trace"]["steps"]:
        extra = f"  best={s['detail'].split('best=')[1]}" if "best=" in s["detail"] else ""
        print(f"   - {s['stage']:<14} ms={s.get('ms')}  {s['detail']}")
    return res


if __name__ == "__main__":
    # 对照：正常命中（路由 retrieve，best>=阈值，0 次重试）
    run("对照-正常命中", "楼上漏水导致我家天花板发霉怎么办", "deepseek")

    # 触发重试：含「猫」治理词 → 走 retrieve；但语料只有遛狗/宠物粪便、无「猫粮」
    # 首检 best=0.1471 < 0.15 → 改写(去「怎么办」) → 复检 0.1757 越阈 → 生成（诚实说明无猫粮依据）
    run("触发重试-宠物离题", "猫主子不吃猫粮怎么办", "deepseek")
