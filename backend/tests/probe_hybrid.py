"""混合检索差异化演示：构造「专有词/法条条号」硬 query，
证明纯向量易漏、BM25 能精确捞回。只比对候选排序，不调 LLM。"""
from __future__ import annotations

from app.config import get_settings
from app.data.ingest import ingest
from app.rag.pipeline import RAGPipeline

HARD = [
    ("民法典第二百八十八条 邻居盖房挡采光 依据哪条", "邻里权属纠纷"),
    ("业主大会 表决 占用公共绿地 合法吗", "公共绿地纠纷"),
    ("楼上漏水 民法典第二百九十六条 怎么维权", "漏水渗水纠纷"),
    ("宠物狗咬人 民法典 饲养动物 损害责任", "宠物纠纷"),
    ("维修基金 使用 业主大会 表决 程序", "物业费纠纷"),
]


def top4(pipe, q):
    ranked, _ = pipe._retrieve(q, "demo", [])
    return [(r["payload"].get("category"), round(r["rerank_score"], 4), r["payload"].get("id")) for r in ranked]


def main():
    s = get_settings()
    ingest(force=True)

    s.enable_hybrid = False
    pv = RAGPipeline(s)
    s.enable_hybrid = True
    ph = RAGPipeline(s)

    for q, exp_cat in HARD:
        v = top4(pv, q)
        h = top4(ph, q)
        v_rank = next((i + 1 for i, x in enumerate(v) if x[0] == exp_cat), None)
        h_rank = next((i + 1 for i, x in enumerate(h) if x[0] == exp_cat), None)
        print(f"\nQ: {q}")
        print(f"  期望类目={exp_cat}")
        print(f"  纯向量 Top4 : {v}")
        print(f"  混合检索 Top4: {h}")
        tag = []
        if v_rank is None:
            tag.append(f"向量未进Top4 ❌")
        if h_rank is not None:
            tag.append(f"混合进Top4(第{h_rank}位) ✅")
        if v_rank and h_rank and h_rank < v_rank:
            tag.append(f"混合排名更靠前({v_rank}→{h_rank}) ⬆")
        print("  =>", "；".join(tag) if tag else "两者一致")


if __name__ == "__main__":
    main()
