"""服务入口（零依赖版，使用标准库 http.server）。

接口：
  GET  /api/health           健康检查 + 知识库条数
  GET  /api/models           可用模型列表
  POST /api/chat             矛盾调解问答 { question, provider?, history?:[{role,content}] }

  GET  /api/kb/stats         知识库统计（总数/已发布/草稿/分块/类别分布）
  GET  /api/kb/docs          文档列表（支持 ?status=&category=&page=&size=）
  POST /api/kb/upload        上传文档（base64 JSON）-> 草稿
  POST /api/kb/import-directory  批量导入目录（files:[{filename,content_base64}]）-> 仅存为草稿
  POST /api/kb/publish-selected  发布选中的草稿（{ids:[...]}，批量嵌入，性能好）
  POST /api/kb/delete-selected   批量删除选中（本地文件+向量库一起删）{ids:[...]}
  POST /api/kb/publish-all        一键发布全部草稿（批量嵌入，性能好）
  GET  /api/kb/{id}/content      读取文档正文（详情预览用，过大文件标记 too_large）
  POST /api/kb/{id}/publish  发布文档（嵌入+入向量库+重建BM25）
  POST /api/kb/{id}/unpublish 下架文档（从向量库移除）
  DELETE /api/kb/{id}        删除文档（含物理文件）

启动时通过 KBManager 确保知识库已就绪（首次自动迁移种子+文件语料并置已发布）。
生产环境可改用 FastAPI 版本（见 README「生产部署」），接口完全一致。

运行：python -m app.main   （默认 0.0.0.0:8000）
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from urllib.parse import urlparse, parse_qs, unquote

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.config import MODEL_REGISTRY, get_settings
from app.log import get_logger, setup_logging
from app.rag.pipeline import RAGPipeline

log = get_logger("app.main")
settings = get_settings()
pipeline = RAGPipeline(settings)
_kb_instance = None


def _kb():
    """知识库管理器单例。"""
    global _kb_instance
    if _kb_instance is None:
        from app.kb import KBManager

        _kb_instance = KBManager(settings)
    return _kb_instance


def _ensure_kb():
    """通过 KBManager 确保知识库就绪：首次自动迁移种子+文件语料并全部置已发布。"""
    try:
        kb = _kb()
        migrated = kb.ensure()
        n = kb.load_published_into_store(reset=migrated)
        # BM25 稀疏索引与向量库强一致（发布/下架会再次触发）
        pipeline.rebuild_bm25()
        st = kb.stats()
        log.info(
            "知识库已就绪 | 文档=%d 已发布=%d 草稿=%d 分块=%d 向量库=%d (首次迁移=%s)",
            st["total"], st["published"], st["draft"], st["chunks"], n, migrated,
        )
    except Exception as e:  # 入库失败不应阻断服务
        log.warning("知识库初始化跳过：%s", e)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict, trace_id: str | None = None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if trace_id:
            self.send_header("X-Trace-Id", trace_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    # ---------- KB 路径解析 ----------
    def _kb_doc_action(self, path: str) -> tuple[str, str] | None:
        """匹配 /api/kb/{id}/publish|unpublish，返回 (doc_id, action)。"""
        parts = [p for p in path.split("/") if p]
        # parts: ['api','kb', {id}, {action}]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "kb" and parts[3] in ("publish", "unpublish"):
            return unquote(parts[2]), parts[3]
        return None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, {"status": "ok", "docs": pipeline.store.count()})
        elif path == "/api/models":
            from app.rag.llm import list_available_models

            self._send(200, {"default": settings.default_llm, "models": list_available_models()})
        elif path == "/api/kb/stats":
            self._send(200, {"status": "ok", "stats": _kb().stats()})
        elif path == "/api/kb/docs":
            qs = parse_qs(urlparse(self.path).query)
            status = qs.get("status", [""])[0]
            category = qs.get("category", [""])[0]
            page = int(qs.get("page", ["1"])[0] or 1)
            size = int(qs.get("size", ["50"])[0] or 50)
            self._send(200, {"status": "ok", **_kb().list_docs(status, category, page, size)})
        # 知识库：读取文档正文（详情预览）
        elif path.startswith("/api/kb/") and path.endswith("/content"):
            parts = [p for p in path.split("/") if p]
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "kb" and parts[3] == "content":
                doc = _kb().get_content(unquote(parts[2]))
                if doc is None:
                    self._send(404, {"error": "文档不存在"})
                    return
                self._send(200, {"status": "ok", **doc})
                return
            self._send(400, {"error": "路径格式错误，应为 /api/kb/{id}/content"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        # 知识库：上传
        if path == "/api/kb/upload":
            fn = (data.get("filename") or "").strip()
            b64 = data.get("content_base64") or ""
            if not fn or not b64:
                self._send(400, {"error": "filename 与 content_base64 必填"})
                return
            try:
                raw_bytes = base64.b64decode(b64)
            except Exception:
                self._send(400, {"error": "content_base64 解码失败"})
                return
            try:
                doc = _kb().add_upload(fn, raw_bytes)
                self._send(200, {"status": "ok", "doc": doc})
            except Exception as e:
                self._send(400, {"error": str(e)})
            return

        # 知识库：批量导入目录（前端逐文件读取后批量提交）
        if path == "/api/kb/import-directory":
            files = data.get("files") or []
            if not isinstance(files, list) or not files:
                self._send(400, {"error": "files 必填（数组：[{filename, content_base64}]）"})
                return
            added = []
            for fobj in files:
                fn = (fobj.get("filename") or "").strip()
                b64 = fobj.get("content_base64") or ""
                if not fn or not b64:
                    continue
                try:
                    raw_bytes = base64.b64decode(b64)
                except Exception:
                    log.warning("导入跳过（解码失败）：%s", fn)
                    continue
                try:
                    doc = _kb().add_upload(fn, raw_bytes)
                    added.append(doc)
                except Exception as e:
                    log.warning("导入跳过 %s：%s", fn, e)
            # 仅存为草稿：不自动发布，待用户在「知识库」页勾选后手动发布进向量库
            self._send(200, {
                "status": "ok",
                "added": len(added),
                "docs": added,
            })
            return

        # 知识库：一键发布全部草稿
        if path == "/api/kb/publish-all":
            drafts = [e["id"] for e in _kb()._index.values() if e["status"] != "published"]
            if not drafts:
                self._send(200, {"status": "ok", "published": 0, "total": 0, "chunks": 0})
                return
            res = _kb().publish_many(drafts)
            self._send(200, {"status": "ok", **res})
            return

        # 知识库：发布选中的草稿（前端勾选后调用）
        if path == "/api/kb/publish-selected":
            ids = data.get("ids") or []
            if not isinstance(ids, list) or not ids:
                self._send(400, {"error": "ids 必填（数组：文档 id 列表）"})
                return
            ids = [str(i) for i in ids]
            res = _kb().publish_many(ids)
            self._send(200, {"status": "ok", **res})
            return

        # 知识库：批量删除选中（本地文件 + 向量库一起删）
        if path == "/api/kb/delete-selected":
            ids = data.get("ids") or []
            if not isinstance(ids, list) or not ids:
                self._send(400, {"error": "ids 必填（数组：文档 id 列表）"})
                return
            ids = [str(i) for i in ids]
            res = _kb().delete_many(ids)
            self._send(200, {"status": "ok", **res})
            return

        # 知识库：发布 / 下架
        doc_action = self._kb_doc_action(path)
        if doc_action:
            doc_id, action = doc_action
            ok = _kb().publish(doc_id) if action == "publish" else _kb().unpublish(doc_id)
            if not ok:
                self._send(404, {"error": f"文档不存在或操作失败：{doc_id}"})
                return
            # BM25 重建已交由 KBManager 内部处理，避免重复重建
            self._send(200, {"status": "ok", "action": action, "doc_id": doc_id})
            return

        # 问答
        if path == "/api/chat":
            question = (data.get("question") or "").strip()
            provider = data.get("provider")
            # 多轮对话：接收前端传来的历史（[{role, content}]），容错限制条数
            history = data.get("history")
            if not isinstance(history, list):
                history = []
            history = history[-12:]  # 兜底截断，pipeline 内部还会再规整
            if not question:
                self._send(400, {"error": "question 不能为空"})
                return
            if provider and provider not in MODEL_REGISTRY:
                provider = None
            req_id = uuid.uuid4().hex[:12]
            t0 = time.perf_counter()
            log.info("[%s] POST /api/chat | provider=%s | 历史=%d | q=%r", req_id, provider or settings.default_llm, len(history), question[:60])
            try:
                result = pipeline.query(question, provider, history=history)
                result["trace_id"] = result.get("trace_id", req_id)
                dt = (time.perf_counter() - t0) * 1000
                log.info("[%s] 响应 | code=200 | route=%s | 耗时=%.0fms", req_id, result["route"], dt)
                self._send(200, result, trace_id=result["trace_id"])
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                log.error("[%s] 处理异常 | code=500 | 耗时=%.0fms | %s", req_id, dt, e)
                self._send(500, {"error": str(e), "trace_id": req_id})
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        # /api/kb/{id}
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "kb":
            doc_id = unquote(parts[2])
            if not _kb().delete(doc_id):
                self._send(404, {"error": f"文档不存在：{doc_id}"})
                return
            # BM25 重建已由 KBManager.delete 内部处理
            self._send(200, {"status": "ok", "action": "delete", "doc_id": doc_id})
            return
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # 静默默认访问日志，统一走 app.main 的 request 日志


def main():
    setup_logging(settings.log_level)
    _ensure_kb()
    port = int(os.environ.get("PORT", settings.port))
    server = ThreadingHTTPServer((settings.host, port), Handler)
    log.info("社区矛盾调解 RAG 助手已启动： http://localhost:%d", port)
    log.info("接口：/api/health  /api/models  /api/chat  /api/kb/*  | 默认模型=%s", settings.default_llm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
