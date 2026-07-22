"""后端冒烟测试（mock 模式，无需任何 API key）。

覆盖：
  1) 检索类问题 → Supervisor 路由到 retrieve + 返回来源
  2) 问候/能力问答 → direct 直答
  3) 过短问题 → clarify 澄清
  4) 知识库无对应 → Self-RAG 重试后仍诚实拒答
"""
from app.data.ingest import ingest
from app.rag.pipeline import RAGPipeline


def main():
    print("== 入库 ==")
    n = ingest(force=True)
    print(f"知识库条数: {n}")

    pipe = RAGPipeline()

    cases = [
        ("楼上漏水导致我家天花板发霉怎么办", "retrieve"),
        ("你好", "direct"),
        ("漏水", "clarify"),
        ("怎么给猫办签证去月球", "retrieve(无依据)"),
    ]

    for q, expect in cases:
        print("\n" + "=" * 60)
        print(f"问: {q}  (期望: {expect})")
        r = pipe.query(q)
        print(f"路由: {r['route']} | 自纠错重试: {r['self_rag_retries']} | 模型: {r['model']}")
        print(f"答案: {r['answer'][:160]}")
        if r["sources"]:
            print(f"来源数: {len(r['sources'])} -> " + " | ".join(s['title'] for s in r['sources'][:3]))


if __name__ == "__main__":
    main()
