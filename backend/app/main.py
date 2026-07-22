"""服务入口（零依赖版，使用标准库 http.server）。

接口：
  GET  /api/health   健康检查 + 知识库条数
  GET  /api/models   可用模型列表
  POST /api/chat     矛盾调解问答 { question, provider? }

启动时自动确保知识库已入库（无数据则写入样例）。
生产环境可改用 FastAPI 版本（见 README「生产部署」），接口完全一致。

运行：python -m app.main   （默认 0.0.0.0:8000）
"""
from __future__ import annotations

import json
import os
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from app.config import MODEL_REGISTRY, get_settings
from app.data.ingest import ingest
from app.log import get_logger, setup_logging
from app.rag.llm import list_available_models
from app.rag.pipeline import RAGPipeline

log = get_logger("app.main")
settings = get_settings()
pipeline = RAGPipeline(settings)


def _ensure_kb():
    try:
        n = ingest(force=False)
        log.info("知识库已就绪，共 %d 条。", n)
    except Exception as e:  # 入库失败不应阻断服务
        log.warning("知识库入库跳过：%s", e)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict, trace_id: str | None = None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if trace_id:
            self.send_header("X-Trace-Id", trace_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, {"status": "ok", "docs": pipeline.store.count()})
        elif path == "/api/models":
            self._send(200, {"default": settings.default_llm, "models": list_available_models()})
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

        if path == "/api/chat":
            question = (data.get("question") or "").strip()
            provider = data.get("provider")
            if not question:
                self._send(400, {"error": "question 不能为空"})
                return
            if provider and provider not in MODEL_REGISTRY:
                provider = None
            req_id = uuid.uuid4().hex[:12]
            t0 = time.perf_counter()
            log.info("[%s] POST /api/chat | provider=%s | q=%r", req_id, provider or settings.default_llm, question[:60])
            try:
                result = pipeline.query(question, provider)
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

    def log_message(self, fmt, *args):
        pass  # 静默默认访问日志，统一走 app.main 的 request 日志


def main():
    setup_logging(settings.log_level)
    _ensure_kb()
    port = int(os.environ.get("PORT", settings.port))
    server = ThreadingHTTPServer((settings.host, port), Handler)
    log.info("社区矛盾调解 RAG 助手已启动： http://localhost:%d", port)
    log.info("接口：/api/health  /api/models  /api/chat  | 默认模型=%s", settings.default_llm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
