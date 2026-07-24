"""文档入库：读取矛盾调解样例 → 向量化 → 写入向量库。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import get_settings
from app.log import get_logger
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import get_vector_store, point_id

log = get_logger("data.ingest")
DATA_PATH = Path(__file__).parent / "mediation_cases.json"
# 入库管道（app/ingest）产出的文档持久化索引，与种子案例合并为统一检索源
INGESTED_JSONL = Path(__file__).parent / "ingested.jsonl"


def _embed_text(doc: dict) -> str:
    return "\n".join(
        str(doc.get(k, "")) for k in ("title", "content", "legal_basis")
    )


def load_documents() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    # 合并入库管道产生的文档（文件入库），作为 BM25 与统一检索源的补充
    if INGESTED_JSONL.exists():
        with open(INGESTED_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def build_points(docs: list[dict], embedder: EmbeddingClient) -> list[dict]:
    texts = [_embed_text(d) for d in docs]
    vectors = embedder.embed(texts)
    points = []
    for doc, vec in zip(docs, vectors):
        points.append(
            {
                "id": point_id(doc["id"]),
                "vector": vec,
                "payload": {
                    "id": doc["id"],
                    "category": doc.get("category", ""),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "legal_basis": doc.get("legal_basis", ""),
                    "mediation_steps": doc.get("mediation_steps", []),
                    "source": doc.get("source", ""),
                },
            }
        )
    return points


def ingest(force: bool = False) -> int:
    settings = get_settings()
    store = get_vector_store(settings)
    if force and store.count() > 0:
        log.info("强制重建知识库，先清空现有 %d 条", store.count())
        store.reset()
    if store.count() > 0 and not force:
        log.info("知识库已存在 %d 条，跳过入库", store.count())
        return store.count()
    docs = load_documents()
    embedder = EmbeddingClient(settings)
    log.info("开始入库 | 文档数=%d | embedding=%s", len(docs), "real" if not embedder.use_mock else "mock")
    t0 = time.perf_counter()
    points = build_points(docs, embedder)
    store.upsert(points)
    log.info("入库完成 | 共 %d 条 | 耗时=%.2fs", store.count(), time.perf_counter() - t0)
    return store.count()


if __name__ == "__main__":
    n = ingest(force=True)
    print(f"已入库 {n} 条矛盾调解知识。")
