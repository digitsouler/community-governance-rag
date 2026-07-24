"""文件加载器：把 PDF / Word / Markdown / 文本 / JSON 解析为纯文本。

重型解析依赖（pypdf / python-docx / paddleocr / pdf2image）按需懒加载，
沙箱无依赖时仍可处理 .md / .txt / .json；其余格式在用户本机
`pip install pypdf python-docx` 后即可用，扫描件另需 `paddleocr pdf2image`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger("ingest.loaders")

SUPPORTED = {".md", ".txt", ".json", ".pdf", ".docx", ".doc"}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> str:
    import json

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for k in ("content", "text", "body", "article"):
            if obj.get(k):
                return str(obj[k])
        return "\n".join(f"{k}：{v}" for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(str(x) for x in obj)
    return str(obj)


def load_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore
    except Exception:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except Exception:
            raise RuntimeError(
                "未安装 PDF 解析库。请在本机执行：pip install pypdf  "
                "（扫描件/图片型 PDF 还需 paddleocr + pdf2image）"
            )
    reader = pypdf.PdfReader(str(path))
    parts = [p.extract_text() or "" for p in reader.pages]
    text = "\n".join(parts).strip()
    if not text:
        return ocr_pdf(path)
    return text


def _load_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception:
        raise RuntimeError("未安装 Word 解析库。请在本机执行：pip install python-docx")
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def ocr_pdf(path: Path) -> str:
    """图片型 PDF：当前抛出明确安装指引，预留外部 OCR 服务接口。"""
    raise RuntimeError(
        "检测到图片型/扫描件 PDF，需要 OCR。请在本机执行："
        "pip install pdf2image paddleocr 并安装 poppler；"
        "或在 config 中配置 OCR_SERVICE_URL 走外部 OCR 服务后重试。"
    )


def load_any(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = load_text(path)
    elif suffix == ".json":
        text = load_json(path)
    elif suffix == ".pdf":
        text = load_pdf(path)
    elif suffix in {".docx", ".doc"}:
        text = _load_docx(path)
    else:
        raise ValueError(f"不支持的文件类型：{suffix}")
    return {"text": text, "meta": {"source": str(path), "ext": suffix}}
