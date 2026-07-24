"""中文友好的文本切分：按段落/句子累积，超长切分并保留尾部重叠。"""
from __future__ import annotations

import re


def _split_long(text: str, max_chars: int) -> list[str]:
    segs = re.split(r"(?<=[。！？；;])", text)
    chunks, buf = [], ""
    for s in segs:
        if len(buf) + len(s) <= max_chars:
            buf += s
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def split_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """把文档切成适合检索的块。

    - 优先按换行/空行保持段落语义完整；
    - 单段落超长时按句切分；
    - 块间保留 overlap 字符重叠，降低边界信息丢失。
    """
    if not text or not text.strip():
        return []
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                chunks.extend(_split_long(p, max_chars))
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for c in chunks[1:]:
            prev = out[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            out.append(f"{tail}\n{c}".strip() if tail else c)
        chunks = out
    return chunks
