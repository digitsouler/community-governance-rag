"""命令行入口：python -m app.ingest --src ./corpus/raw [--reset] [--tenant-id x]

示例：
  # 默认扫描 backend/corpus/raw，增量入库
  PYTHONPATH=. python -m app.ingest

  # 全量重建（先清空向量库与索引）
  PYTHONPATH=. python -m app.ingest --src ../corpus/raw --reset

  # 多租户隔离（预留，向量库按 payload.tenant_id 过滤即可）
  PYTHONPATH=. python -m app.ingest --tenant-id acme
"""
import argparse

from app.ingest.pipeline import run_ingest


def main() -> None:
    ap = argparse.ArgumentParser(description="社区治理 RAG 入库管道")
    ap.add_argument("--src", default="corpus/raw", help="语料根目录（递归扫描支持的格式）")
    ap.add_argument("--reset", action="store_true", help="清空后全量重建")
    ap.add_argument("--tenant-id", default="", help="多租户隔离标识（预留）")
    args = ap.parse_args()
    n = run_ingest(args.src, reset=args.reset, tenant_id=args.tenant_id)
    print(f"入库完成，知识库当前共 {n} 条。重启后端服务即可让 BM25 与检索生效。")


if __name__ == "__main__":
    main()
