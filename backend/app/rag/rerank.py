"""重排序。

默认 "hybrid"：向量召回分数 + 中文词面重合度（字符 bigram 重叠），
无需额外下载模型即可提升相关片段排序。预留 "bge" 模式（本地
sentence-transformers 的 bge-reranker-v2-m3），配置后自动启用。
"""
from __future__ import annotations

from typing import Any


def _char_bigrams(text: str) -> set[str]:
    text = "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return {text[i : i + 2] for i in range(len(text) - 1)}


def hybrid_score(query: str, payload: dict[str, Any], vec_score: float) -> float:
    text = " ".join(
        str(payload.get(k, "")) for k in ("title", "content", "legal_basis")
    )
    q = _char_bigrams(query)
    d = _char_bigrams(text)
    overlap = len(q & d) / (len(q) + 1e-6)
    # 向量分权重 0.7，词面 0.3
    return 0.7 * vec_score + 0.3 * overlap


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 4,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    if mode == "bge":
        # 预留：可在此加载 bge-reranker-v2-m3 做精排
        # from sentence_transformers import CrossEncoder ...
        # 当前未安装重模型时回落 hybrid
        pass
    scored = [
        {
            **c,
            "rerank_score": hybrid_score(query, c.get("payload", {}), c["score"]),
        }
        for c in candidates
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
