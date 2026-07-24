"""入库管道包。

把 PDF / Word / Markdown / 文本 / JSON 等真实文档源，解析 → 切分 → 向量化
→ 增量写入向量库（默认内存库，配置 qdrant_url 后自动切 Qdrant）。

设计要点：
- 场景无关：与具体业务（矛盾调解）解耦，换场景只换语料目录。
- 增量 upsert：按 doc_id 去重，重跑只补新文档，不重复烧 embedding。
- 多租户预留：payload 带 tenant_id 字段，向量库可后续按 namespace 隔离。
- 重型依赖懒加载：沙箱无 pypdf/python-docx 也能处理 .md/.txt/.json。
"""
from app.ingest.pipeline import run_ingest

__all__ = ["run_ingest"]
