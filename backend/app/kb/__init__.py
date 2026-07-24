"""知识库管理：文档生命周期（草稿 → 发布 → 下架 → 删除）。

设计要点（文件即数据源，不再内联全文到 JSON）：
  - 单一事实来源是磁盘上的文件：
      * ``corpus/docs/``    —— 种子/样例知识库（含 seeds/ 子目录），默认已发布；
      * ``corpus/uploads/`` —— 用户上传/导入的文件，默认草稿，审核后发布。
  - 轻量索引 ``kb_index.json`` 只记录每篇文档的元数据（id / 类型 / 文件路径 /
    状态 / 标题 / 类别 / 分块数），**不存放正文**；正文在发布时按需从文件解析，
    因此索引不会随语料增长而膨胀。
  - 「已发布」文档才进入向量库（稠密）+ BM25（稀疏），即检索可见；
    草稿只存在于文件系统 + 索引，不进检索，便于先审核再上架。
  - 每个分块 payload 携带 doc_id，支持按文档整体上下架（delete_by_doc_id）。
  - 首次启动（索引为空）自动扫描两个目录登记文档，并全部发布种子知识。

对外：KBManager。后端 API 与前端知识库后台均围绕它构建。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.ingest.loaders import SUPPORTED, load_any
from app.ingest.pipeline import normalize_chunk, split_text
from app.log import get_logger
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import get_vector_store, point_id

log = get_logger("kb")

# 种子/样例知识库（文件即数据，默认发布）
SEED_DIR = Path(__file__).parent.parent.parent / "corpus" / "docs"
# 用户上传 / 导入（默认草稿）
UPLOAD_DIR = Path(__file__).parent.parent.parent / "corpus" / "uploads"
# 轻量索引：仅元数据，不含正文
INDEX_PATH = Path(__file__).parent.parent / "data" / "kb_index.json"

SEED_PREFIX = "file:"
UPLOAD_PREFIX = "upload:"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w一-鿿.\-]", "_", name)
    return name or "doc"


def _iter_files(src: Path) -> list[Path]:
    if not src.exists():
        return []
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


class KBManager:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    # ---------- 索引持久化（仅元数据） ----------
    def _load_index(self) -> dict[str, dict]:
        if INDEX_PATH.exists():
            try:
                return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        INDEX_PATH.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- 文件解析（按需，发布时调用） ----------
    def _parse_file(self, path: Path) -> dict[str, Any]:
        """读取文件 → 切分 → 归一化分块。返回 {chunks, count}。"""
        res = load_any(path)
        text = res.get("text", "")
        chunks_raw = split_text(text, self.s.chunk_size, self.s.chunk_overlap)
        chunks = [normalize_chunk(path.stem, ch, i, res.get("meta", {})) for i, ch in enumerate(chunks_raw)]
        return {"chunks": chunks, "count": len(chunks)}

    def _register(self, path: Path, kind: str) -> dict | None:
        """从文件解析元数据并登记一篇文档到索引。返回索引条目或 None（空文件）。"""
        try:
            meta = self._parse_file(path)
        except Exception as e:
            log.warning("KB 解析跳过 %s：%s", path, e)
            return None
        if meta["count"] == 0:
            return None
        doc_id = (SEED_PREFIX if kind == "seed" else UPLOAD_PREFIX) + path.stem
        first = meta["chunks"][0]
        entry = {
            "id": doc_id,
            "type": "file" if kind == "seed" else "upload",
            "file_path": str(path),
            "status": "published" if kind == "seed" else "draft",
            "title": first.get("title") or path.stem,
            "category": first.get("category", ""),
            "source": first.get("source", "") or f"用户上传/{path.name}",
            "chunk_count": meta["count"],
            "created_at": _now(),
        }
        return entry

    # ---------- 迁移 / 扫描（首次启动） ----------
    def ensure(self) -> bool:
        """扫描两个目录登记文档。

        首次运行（索引为空）时，种子知识默认置为 published，返回 True 触发向量库重建；
        已运行过的启动则增量登记新文件、清理已被手动删除的文件。

        返回是否执行了迁移（True 表示需要强制重建向量库以统一 payload）。
        """
        migrated = not self._index

        # 1) 种子知识库：corpus/docs（含子目录），默认发布
        for f in _iter_files(SEED_DIR):
            doc_id = SEED_PREFIX + f.stem
            if doc_id in self._index:
                continue
            entry = self._register(f, "seed")
            if entry:
                self._index[doc_id] = entry

        # 2) 用户上传/导入：corpus/uploads，默认草稿
        for f in _iter_files(UPLOAD_DIR):
            doc_id = UPLOAD_PREFIX + f.stem
            if doc_id in self._index:
                continue
            entry = self._register(f, "upload")
            if entry:
                self._index[doc_id] = entry

        # 3) 清理索引中文件已被删除的条目（同步移除其在向量库的点）
        removed = [
            d for d, e in self._index.items()
            if e.get("file_path") and not Path(e["file_path"]).exists()
        ]
        if removed:
            store = get_vector_store(self.s)
            for doc_id in removed:
                try:
                    if self._index[doc_id].get("status") == "published":
                        store.delete_by_doc_id(doc_id)
                except Exception:
                    pass
                del self._index[doc_id]
            log.info("清理 %d 篇索引中文件已删除的文档", len(removed))

        self._save_index()
        return migrated

    # ---------- 载入向量库 ----------
    def load_published_into_store(self, reset: bool = False) -> int:
        """把 published 文档从文件解析、嵌入并 upsert 进向量库。返回库内总条数。"""
        store = get_vector_store(self.s)
        if reset and store.count() > 0:
            log.info("强制重建知识库，先清空现有 %d 条", store.count())
            store.reset()
        if store.count() > 0 and not reset:
            return store.count()
        emb = EmbeddingClient(self.s)
        published = [e for e in self._index.values() if e["status"] == "published"]
        points = []
        for e in published:
            path = Path(e["file_path"])
            if not path.exists():
                continue
            chunks = self._parse_file(path)["chunks"]
            for i, c in enumerate(chunks):
                c["id"] = f"{path.stem}:{i}"
                c["doc_id"] = e["id"]
            texts = [f"{c['title']}\n{c['content']}\n{c['legal_basis']}" for c in chunks]
            vecs = emb.embed(texts)
            for c, vec in zip(chunks, vecs):
                points.append({"id": point_id(c["id"]), "vector": vec, "payload": {**c}})
        if points:
            store.upsert(points)
            log.info("入库完成 | 已发布文档=%d 分块=%d", len(published), len(points))
        return store.count()

    # ---------- 查询 ----------
    def list_docs(self, status: str = "", category: str = "", page: int = 1, size: int = 50) -> dict:
        items = list(self._index.values())
        if status:
            items = [d for d in items if d["status"] == status]
        if category:
            items = [d for d in items if d.get("category") == category]
        items.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        total = len(items)
        start = max(0, (page - 1) * size)
        page_items = items[start: start + size]
        return {"total": total, "page": page, "size": size, "items": page_items}

    def stats(self) -> dict:
        cats: dict[str, int] = {}
        published = draft = 0
        for d in self._index.values():
            if d["status"] == "published":
                published += 1
            else:
                draft += 1
            cats[d.get("category") or "其他"] = cats.get(d.get("category") or "其他", 0) + 1
        return {
            "total": len(self._index),
            "published": published,
            "draft": draft,
            "chunks": sum(d.get("chunk_count", 0) for d in self._index.values()),
            "categories": cats,
        }

    def get(self, doc_id: str) -> dict | None:
        return self._index.get(doc_id)

    # 详情预览：读取文件正文（不大时直接返回，便于前端弹窗查看）
    MAX_PREVIEW_CHARS = 20000   # 提取文本超过此长度视为过大，不返回正文
    MAX_PREVIEW_BYTES = 2 * 1024 * 1024  # 源文件超过 2MB 视为过大

    def get_content(self, doc_id: str) -> dict | None:
        """读取文档正文用于详情预览。

        返回 {id, title, category, status, source, size, too_large, text}。
        too_large=True 时 text 为空，前端提示文件过大无法预览。
        """
        entry = self._index.get(doc_id)
        if not entry:
            return None
        path = Path(entry["file_path"])
        if not path.exists():
            return {"id": doc_id, "title": entry.get("title", ""), "too_large": True,
                    "missing": True, "text": "", "size": 0}
        size = path.stat().st_size
        if size > self.MAX_PREVIEW_BYTES:
            return {"id": doc_id, "title": entry.get("title", ""), "size": size,
                    "too_large": True, "text": "", "missing": False}
        try:
            res = load_any(path)
            text = res.get("text", "")
            if len(text) > self.MAX_PREVIEW_CHARS:
                return {"id": doc_id, "title": entry.get("title", ""), "size": size,
                        "too_large": True, "text": "", "missing": False}
            return {
                "id": doc_id,
                "title": entry.get("title", ""),
                "category": entry.get("category", ""),
                "status": entry.get("status", ""),
                "source": entry.get("source", ""),
                "size": size,
                "too_large": False,
                "missing": False,
                "text": text,
            }
        except Exception as e:
            log.warning("读取正文失败 %s：%s", doc_id, e)
            return {"id": doc_id, "title": entry.get("title", ""), "size": size,
                    "too_large": True, "text": "", "missing": False, "error": str(e)}

    # ---------- 生命周期 ----------
    def add_upload(self, filename: str, raw_bytes: bytes) -> dict:
        """保存上传文件 → 解析 → 登记为 draft。返回文档元信息。"""
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(_safe_name(filename)).stem
        suffix = Path(filename).suffix
        path = UPLOAD_DIR / f"{stem}{suffix}"
        if path.exists():
            path = UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        path.write_bytes(raw_bytes)
        entry = self._register(path, "upload")
        if entry is None:
            path.unlink(missing_ok=True)
            raise RuntimeError("文件内容为空或无法解析（扫描件需 OCR 后重试）")
        # 同名防冲突
        while entry["id"] in self._index:
            entry["id"] = f"{UPLOAD_PREFIX}{path.stem}_{uuid.uuid4().hex[:6]}"
            entry["file_path"] = str(path)
        self._index[entry["id"]] = entry
        self._save_index()
        return {
            "id": entry["id"],
            "title": entry["title"],
            "category": entry["category"],
            "chunk_count": entry["chunk_count"],
            "status": entry["status"],
        }

    def publish(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        if not entry:
            return False
        if entry["status"] == "published":
            return True
        path = Path(entry["file_path"])
        if not path.exists():
            log.warning("文件不存在，无法发布：%s", path)
            return False
        chunks = self._parse_file(path)["chunks"]
        for i, c in enumerate(chunks):
            c["id"] = f"{path.stem}:{i}"
            c["doc_id"] = doc_id
        emb = EmbeddingClient(self.s)
        store = get_vector_store(self.s)
        texts = [f"{c['title']}\n{c['content']}\n{c['legal_basis']}" for c in chunks]
        vecs = emb.embed(texts)
        points = [
            {"id": point_id(c["id"]), "vector": vec, "payload": {**c}}
            for c, vec in zip(chunks, vecs)
        ]
        store.upsert(points)
        entry["status"] = "published"
        entry["chunk_count"] = len(chunks)
        entry["title"] = chunks[0].get("title") or path.stem
        entry["category"] = chunks[0].get("category", "")
        self._save_index()
        self._rebuild_bm25()
        log.info("文档发布 | id=%s 分块=%d", doc_id, len(chunks))
        return True

    def publish_many(self, doc_ids: list[str]) -> dict:
        """批量发布：把所有待发布文档的分块合并后一次性嵌入 + 入向量库，
        BM25 仅重建一次。相比逐个 publish 大幅减少 embedding 网络往返与
        BM25 重建次数，是目录导入/一键发布草稿的性能关键路径。
        """
        store = get_vector_store(self.s)
        emb = EmbeddingClient(self.s)
        collected: list[tuple[str, list[dict]]] = []  # (doc_id, chunks)
        for doc_id in doc_ids:
            entry = self._index.get(doc_id)
            if not entry or entry["status"] == "published":
                continue
            path = Path(entry["file_path"])
            if not path.exists():
                log.warning("批量发布跳过（文件缺失）：%s", doc_id)
                continue
            try:
                chunks = self._parse_file(path)["chunks"]
            except Exception as e:
                log.warning("批量发布跳过（解析失败）：%s | %s", doc_id, e)
                continue
            for i, c in enumerate(chunks):
                c["id"] = f"{path.stem}:{i}"
                c["doc_id"] = doc_id
            collected.append((doc_id, chunks))

        if not collected:
            return {"published": 0, "total": len(doc_ids), "chunks": 0}

        # 合并全部文本，单次 embed（内部按 BATCH 分批），减少网络往返
        flat: list[str] = []
        meta: list[tuple[str, dict]] = []
        for doc_id, chunks in collected:
            for c in chunks:
                flat.append(f"{c['title']}\n{c['content']}\n{c['legal_basis']}")
                meta.append((doc_id, c))
        vecs = emb.embed(flat)
        points = [
            {"id": point_id(c["id"]), "vector": vec, "payload": {**c}}
            for (_, c), vec in zip(meta, vecs)
        ]
        store.upsert(points)

        for doc_id, chunks in collected:
            e = self._index[doc_id]
            e["status"] = "published"
            e["chunk_count"] = len(chunks)
            e["title"] = chunks[0].get("title") or Path(e["file_path"]).stem
            e["category"] = chunks[0].get("category", "")
        self._save_index()
        self._rebuild_bm25()
        log.info("批量发布完成 | 文档=%d 分块=%d", len(collected), len(points))
        return {"published": len(collected), "total": len(doc_ids), "chunks": len(points)}

    def unpublish(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        if not entry:
            return False
        if entry["status"] != "published":
            return True
        store = get_vector_store(self.s)
        store.delete_by_doc_id(doc_id)
        entry["status"] = "draft"
        self._save_index()
        self._rebuild_bm25()
        log.info("文档下架 | id=%s", doc_id)
        return True

    def delete(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        if not entry:
            return False
        self.unpublish(doc_id)
        path = Path(entry.get("file_path", ""))
        if path.exists():
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                log.warning("删除源文件失败：%s", e)
        del self._index[doc_id]
        self._save_index()
        return True

    def delete_many(self, doc_ids: list[str]) -> dict:
        """批量删除：逐个 unpublish（向量库）+ 删本地文件 + 清索引，BM25 只重建一次。"""
        deleted = 0
        failed = 0
        for doc_id in doc_ids:
            try:
                ok = self.delete(doc_id)
                if ok:
                    deleted += 1
                else:
                    failed += 1
            except Exception as e:
                log.warning("批量删除跳过 %s：%s", doc_id, e)
                failed += 1
        # BM25 在每次 delete 内部已通过 unpublish 触发 _rebuild_bm25，
        # 但逐次调用效率低；这里统一收尾一次确保一致性
        try:
            self._rebuild_bm25()
        except Exception:
            pass
        return {"deleted": deleted, "failed": failed}

    # ---------- BM25 同步 ----------
    def _rebuild_bm25(self) -> None:
        try:
            from app.rag.pipeline import get_pipeline

            get_pipeline(self.s).rebuild_bm25()
        except Exception as e:  # noqa: BLE001
            log.warning("BM25 重建失败：%s", e)
