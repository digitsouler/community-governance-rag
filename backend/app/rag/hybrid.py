"""稀疏检索（BM25）+ 多路召回融合（RRF）。

为什么需要它：
  稠密向量召回擅长「语义相似」（如「楼上滴水」↔「天花板渗水」），但对
  **字面/专有词**不敏感——矛盾调解场景高频出现的「押金」「维修基金」
  「民法典第×条」「物业费滞纳金」等，向量容易漏，而 BM25 能精确命中。

  BM25 与向量召回互补，两者经 Reciprocal Rank Fusion（RRF）融合后，
  再交给 rerank 精排，既补召回又不破坏已有好排序。

零依赖设计：
  中文用「字符 bigram」分词（无需 jieba）；英文 / 数字连续片段作为整体
  token 保留，确保法条条号、金额等可被精确匹配。
"""
from __future__ import annotations

import math
import re
from typing import Any

_K1 = 1.5  # BM25 词频饱和参数
_B = 0.75  # BM25 文档长度归一化参数


def _tokenize(text: str) -> list[str]:
    """中文→字符 bigram；英文/数字连续片段→整体 token。"""
    text = (text or "").lower()
    tokens: list[str] = []
    # 英文 / 数字连续片段（法条条号、金额、专有名词等）
    tokens.extend(re.findall(r"[a-z0-9]+", text))
    # 中文字符 bigram
    zh = "".join(ch for ch in text if "一" <= ch <= "鿿")
    for i in range(len(zh) - 1):
        tokens.append(zh[i : i + 2])
    return tokens


class BM25Index:
    """零依赖 BM25 索引。build 一次，search 多次。"""

    def __init__(self) -> None:
        self._docs: list[dict] = []  # [{id, payload, tokens}]
        self._df: dict[str, int] = {}
        self._avgdl = 0.0
        self._n = 0
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def build(self, docs: list[dict]) -> None:
        """docs: [{id, payload}]，payload 含 title/content/legal_basis 等文本字段。"""
        self._docs = []
        df: dict[str, int] = {}
        total_len = 0
        for d in docs:
            payload = d.get("payload", d)
            text = " ".join(str(payload.get(k, "")) for k in ("title", "content", "legal_basis"))
            toks = _tokenize(text)
            self._docs.append({"id": d["id"], "payload": payload, "tokens": toks})
            total_len += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._n = len(self._docs)
        self._avgdl = total_len / self._n if self._n else 0.0
        self._built = True

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self._built or not self._docs:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        results: list[tuple[float, dict]] = []
        for d in self._docs:
            score = 0.0
            dl = len(d["tokens"])
            # 预统计该文档的 term 频率
            tf_map: dict[str, int] = {}
            for t in d["tokens"]:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t in q_toks:
                if t not in self._df:
                    continue
                tf = tf_map.get(t, 0)
                if tf == 0:
                    continue
                idf = math.log((self._n - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1.0)
                score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * (dl / (self._avgdl or 1))))
            if score > 0:
                results.append((score, d))
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["id"], "score": sc, "payload": d["payload"]}
            for sc, d in results[:top_k]
        ]


def rrf_fuse(lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion：把多路召回按「排名」融合，不依赖各自分数尺度。

    经典 RRF 公式：score = Σ 1/(k + rank)。rank 从 0 起。k 默认 60。
    返回融合后的候选列表（按融合分降序），并保留每条的 dense 向量分（score）
    供后续 rerank 使用；仅被稀疏召回命中的文档 dense 分记为 0。
    """
    fused: dict[Any, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            doc_id = item["id"]
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "payload": item["payload"],
                    "score": item.get("score", 0.0),  # dense 余弦分
                    "_rrf": 0.0,
                }
            fused[doc_id]["_rrf"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda x: x["_rrf"], reverse=True)
    return ranked
