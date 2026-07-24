"""一次性迁移：把知识库从「JSON manifest 内联全文」改为「文件即数据源」。

做什么：
  1. corpus/raw/*  ->  corpus/docs/*            （160 篇语料以文件形式保留，不再内联进 JSON）
  2. mediation_cases.json 的 48 篇种子 -> corpus/docs/seeds/*.md（带 frontmatter，可被 load_any 解析）
  3. 删除 kb_manifest.jsonl、ingested.jsonl（体积大、与文件重复）

迁移后 KBManager 以 corpus/docs/（种子知识，默认发布）+ corpus/uploads/（用户文件，默认草稿）
为唯一数据源，元数据仅存轻量 kb_index.json（不含正文）。

运行：PYTHONPATH=. python scripts/migrate_kb_to_files.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # backend/
CORPUS_RAW = ROOT / "corpus" / "raw"
CORPUS_DOCS = ROOT / "corpus" / "docs"
SEED_JSON = ROOT / "app" / "data" / "mediation_cases.json"
DATA = ROOT / "app" / "data"


def main() -> None:
    # 1. corpus/raw -> corpus/docs
    CORPUS_DOCS.mkdir(parents=True, exist_ok=True)
    if CORPUS_RAW.exists():
        moved = 0
        for item in sorted(CORPUS_RAW.iterdir()):
            dest = CORPUS_DOCS / item.name
            if dest.exists():
                continue
            shutil.move(str(item), str(dest))
            moved += 1
        # 删除空壳目录
        try:
            CORPUS_RAW.rmdir()
        except OSError:
            pass
        print(f"[1] corpus/raw -> corpus/docs 完成（移动 {moved} 项）")
    else:
        print("[1] corpus/raw 不存在，跳过")

    # 2. 生成 48 篇种子 .md -> corpus/docs/seeds/
    #    类别对齐到语料库风格（如「邻里噪音纠纷」->「噪音扰民」），保证统计类别一致
    CATEGORY_MAP = {
        "邻里噪音纠纷": "噪音扰民",
        "漏水渗水纠纷": "漏水渗水",
        "宠物纠纷": "宠物管理",
        "停车占位纠纷": "停车车位",
        "物业费纠纷": "物业费",
        "婚姻家庭纠纷": "婚姻情感",
        "邻里采光通风纠纷": "装修违建",
        "装修扰民纠纷": "装修违建",
        "环境卫生纠纷": "邻里琐事",
        "邻里权属纠纷": "邻里侵权",
        "租赁纠纷": "租赁纠纷",
        "养老赡养纠纷": "赡养抚养",
        "占道经营纠纷": "邻里琐事",
        "未成年人纠纷": "其他",
        "公共绿地纠纷": "公共设施",
        "家庭暴力纠纷": "家庭暴力",
    }
    seeds_dir = CORPUS_DOCS / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(SEED_JSON.read_text(encoding="utf-8"))
    written = 0
    for c in cases:
        cid = c.get("id") or f"seed_{written+1:03d}"
        cat = CATEGORY_MAP.get(c.get("category", ""), c.get("category", "其他"))
        steps = c.get("mediation_steps", []) or []
        steps_line = " | ".join(steps) if steps else ""
        fm = ["---", f"category: {cat}", f"title: {c.get('title', '')}"]
        if c.get("source"):
            fm.append(f"source: {c['source']}")
        if c.get("legal_basis"):
            fm.append(f"legal_basis: {c['legal_basis']}")
        if steps_line:
            fm.append(f"steps: {steps_line}")
        fm.append("---")
        body = c.get("content", "") or ""
        md = "\n".join(fm) + "\n\n" + body + "\n"
        (seeds_dir / f"{cid}.md").write_text(md, encoding="utf-8")
        written += 1
    print(f"[2] 生成 {written} 篇种子 .md -> corpus/docs/seeds/（类别已对齐）")

    # 3. 删除内联全文的 giant JSON
    for jf in ("kb_manifest.jsonl", "ingested.jsonl"):
        p = DATA / jf
        if p.exists():
            p.unlink()
            print(f"[3] 删除 {jf}（{p.stat().st_size // 1024} KB）")

    print("\n迁移完成。下一步：重启后端，KBManager 会以文件为数据源重建 kb_index.json 并装载向量库。")


if __name__ == "__main__":
    main()
