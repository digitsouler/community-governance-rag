"""入库管道：文件 → 解析 → 切分 → 归一化 → 向量化 → 增量 upsert。

同时把归一化后的文档落到 data/ingested.jsonl，供 BM25 稀疏索引与服务端统一
检索源复用（见 app/data/ingest.load_documents）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.data.ingest import INGESTED_JSONL
from app.ingest.loaders import SUPPORTED, load_any
from app.ingest.split import split_text
from app.log import get_logger
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import get_vector_store, point_id

log = get_logger("ingest.pipeline")

# 类别关键词表：用于从正文自动推断矛盾类别（无 frontmatter 时兜底）
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "噪音扰民": ["噪音", "噪声", "吵", "广场舞", "施工声"],
    "漏水渗水": ["漏水", "渗水", "水管", "天花板", "发霉", "积水"],
    "停车车位": ["停车", "车位", "地锁", "占道", "车库"],
    "宠物管理": ["宠物", "狗", "猫", "犬", "粪便", "咬人"],
    "物业费": ["物业费", "物业", "维修基金", "业委会", "业主大会"],
    "装修违建": ["装修", "违建", "搭建", "采光", "通风", "承重"],
    "邻里琐事": ["油烟", "垃圾", "阳台", "晾衣", "杂物", "堆放"],
    "公共设施": ["电梯", "路灯", "充电桩", "绿地", "健身", "楼道"],
    "租赁纠纷": ["租房", "租客", "房东", "租户", "押金", "涨租"],
    "赡养抚养": ["赡养", "抚养", "老人", "扶养", "赡养费"],
    "家庭暴力": ["家暴", "暴力", "殴打", "人身安全"],
    "婚姻情感": ["婚姻", "离婚", "出轨", "抚养权", "财产分割"],
    "邻里侵权": ["侵权", "隐私", "监控", "摄像头", "偷拍"],
    "消费维权": ["消费", "维权", "商家", "退款", "假货"],
    "劳动社保": ["劳动", "工资", "社保", "工伤", "欠薪"],
    "政策办事": ["政策", "办事", "补贴", "低保", "证明", "落户"],
}
DEFAULT_CATEGORY = "其他"


def detect_category(text: str) -> str:
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in text for k in kws):
            return cat
    return DEFAULT_CATEGORY


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    fm: dict[str, str] = {}
    body = text
    if text.lstrip().startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            body = text[end + 4:].strip()
    return fm, body


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def normalize_chunk(stem: str, chunk: str, idx: int, meta: dict, tenant_id: str = "") -> dict:
    fm, body = _parse_frontmatter(chunk)
    category = fm.get("category") or detect_category(body)
    title = fm.get("title") or _first_heading(body) or stem
    steps = [s.strip() for s in fm.get("steps", "").split("|") if s.strip()] if fm.get("steps") else []
    doc_id = f"{tenant_id + ':' if tenant_id else ''}{stem}:{idx}"
    return {
        "id": doc_id,
        "category": category,
        "title": title,
        "content": body,
        "legal_basis": fm.get("legal_basis", ""),
        "mediation_steps": steps,
        "source": meta.get("source", ""),
        "tenant_id": tenant_id,
    }


def iter_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def run_ingest(src: str | Path, reset: bool = False, tenant_id: str = "",
               settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    store = get_vector_store(settings)
    embedder = EmbeddingClient(settings)
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"语料目录不存在：{src}")

    files = iter_files(src)
    log.info("扫描到 %d 个可入库文件 | src=%s", len(files), src)

    # 载入已有 ingested 索引：用于增量跳过 + 持久化
    index: dict[str, dict] = {}
    if INGESTED_JSONL.exists() and not reset:
        for line in INGESTED_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                index[d["id"]] = d

    docs: list[dict] = []
    for f in files:
        try:
            res = load_any(f)
        except Exception as e:
            log.warning("跳过 %s：%s", f, e)
            continue
        text = res["text"]
        if not text.strip():
            continue
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, ch in enumerate(chunks):
            doc = normalize_chunk(f.stem, ch, i, res["meta"], tenant_id)
            if doc["id"] in index:
                continue
            docs.append(doc)

    if not docs:
        log.info("没有新文档需要入库（全部已存在）。如需全量重建请加 --reset。")
        return len(index)

    log.info("开始向量化 | 新文档块=%d | embedding=%s", len(docs),
             "real" if not embedder.use_mock else "mock")
    t0 = time.perf_counter()
    texts = [f"{d['title']}\n{d['content']}\n{d['legal_basis']}" for d in docs]
    vectors = embedder.embed(texts)
    points = [
        {"id": point_id(d["id"]), "vector": vec, "payload": d}
        for d, vec in zip(docs, vectors)
    ]
    if reset:
        store.reset()
        index = {}
    store.upsert(points)
    index.update({d["id"]: d for d in docs})
    _dump_index(index)
    log.info("入库完成 | 新增=%d 总计=%d | 耗时=%.2fs",
             len(docs), len(index), time.perf_counter() - t0)
    return len(index)


def _dump_index(index: dict) -> None:
    INGESTED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with INGESTED_JSONL.open("w", encoding="utf-8") as f:
        for d in index.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
