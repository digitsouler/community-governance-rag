"""服务入口（零依赖版，使用标准库 http.server）。

接口：
  GET  /api/health           健康检查 + 知识库条数
  GET  /api/models           可用模型列表
  POST /api/chat             矛盾调解问答 { question, provider?, session_id?, history? }
  POST /api/sessions         新建会话（返回 session_id）
  GET  /api/sessions         会话列表（按更新时间倒序，含首问摘要）
  GET  /api/sessions/{id}    会话完整消息
  DELETE /api/sessions/{id}  删除会话

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
from app.profile_store import get_profile_store
from app.user_memory import get_user_memory
from app.session_store import get_session_store
from app.gateway import (
    resolve_user,
    get_ratelimiter,
    get_guardrails,
    get_semantic_cache,
    audit,
    error_payload,
)

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

    def _send_429(self, msg: str, trace_id: str, retry_after: int) -> None:
        body = json.dumps(error_payload(msg, trace_id, 429), ensure_ascii=False).encode("utf-8")
        self.send_response(429)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Retry-After", str(retry_after))
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
        # 会话：列表（仅返回当前用户自己的会话）
        elif path == "/api/sessions":
            user_id = resolve_user(self.headers, self.client_address[0])
            items = get_session_store().list_sessions(owner=user_id)
            self._send(200, {"status": "ok", "sessions": items})
        # 会话：完整消息
        elif path.startswith("/api/sessions/") and len([p for p in path.split("/") if p]) == 3:
            sid = unquote(path.split("/")[3])
            user_id = resolve_user(self.headers, self.client_address[0])
            sess = get_session_store().get_session(sid)
            if sess is None or (sess.get("owner") and sess.get("owner") != user_id):
                self._send(404, {"error": "会话不存在"})
                return
            self._send(200, {"status": "ok", "session": sess,
                             "messages": get_session_store().get_messages(sid)})
        # 会话：结构化案件档案（长期记忆）
        # B 方案：优先按 (user_id, role) 查用户级长期记忆；role 缺省时回退 sid-keyed（兼容旧版）
        elif path.startswith("/api/sessions/") and path.endswith("/profile"):
            sid = unquote(path.split("/")[3])
            user_id = resolve_user(self.headers, self.client_address[0])
            sess = get_session_store().get_session(sid)
            if sess is None or (sess.get("owner") and sess.get("owner") != user_id):
                self._send(404, {"error": "会话不存在"})
                return
            qs = parse_qs(urlparse(self.path).query)
            role = (qs.get("role", [""])[0] or "").strip()
            if role:
                prof = get_user_memory().get_profile(user_id, role)
            else:
                prof = get_profile_store().get(sid)
            self._send(200, {"status": "ok", "session_id": sid, "profile": prof})
        else:
            self._send(404, {"error": "not found"})

    def _stream_chat(self, session_id, question, provider, history, req_id, user_id, ip, has_pii, user_role=None):
        """SSE 流式问答：逐事件推送 route / delta / done / error，首字即可见。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Trace-Id", req_id)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        store = get_session_store()
        gr = get_guardrails()

        def emit(ev: dict):
            try:
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] SSE 写出失败: %s", req_id, e)

        try:
            for ev in pipeline.query(question, provider, history=history, session_id=session_id, stream=True, user_role=user_role or None, user_id=user_id):
                if ev.get("type") == "delta":
                    emit(ev)
                elif ev.get("type") == "route":
                    emit(ev)
                elif ev.get("type") == "done":
                    res = ev["result"]
                    res["trace_id"] = res.get("trace_id", req_id)
                    res["session_id"] = session_id
                    answer = gr.redact(res["answer"])  # ⑤ 输出护栏：PII 脱敏
                    get_semantic_cache().put(question, answer, has_pii)  # ④ 回填语义缓存
                    # B 方案：写入用户级 question_log（跨 session/跨角色汇总）
                    try:
                        get_user_memory().record_question(user_id, session_id, user_role or "", question)
                    except Exception as _e:
                        log.warning("[%s] question_log 写入失败: %s", req_id, _e)
                    emit({
                        "type": "done",
                        "answer": answer,
                        "route": res["route"],
                        "sources": res["sources"],
                        "self_rag_retries": res["self_rag_retries"],
                        "model": res["model"],
                        "latency_ms": res["latency_ms"],
                        "case_profile": res["case_profile"],
                        "trace": res["trace"],
                        "trace_id": res["trace_id"],
                        "session_id": session_id,
                        "decomposition": res.get("decomposition", {"enabled": False, "sub_queries": []}),
                    })
                    store.append_message(session_id, "assistant", answer)
                    audit("chat", trace_id=req_id, user_id=user_id, ip=ip, session_id=session_id,
                          route=res["route"], model=res["model"], latency_ms=res["latency_ms"], stream=True)
            emit({"type": "end"})
        except Exception as e:  # noqa: BLE001
            log.error("[%s] SSE 处理异常 | %s", req_id, e)
            emit({"type": "error", "error": str(e), "trace_id": req_id})
        finally:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads((raw or b"{}").decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        # 会话：新建（记录归属用户）
        if path == "/api/sessions":
            user_id = resolve_user(self.headers, self.client_address[0])
            sid = get_session_store().create_session(owner=user_id)
            self._send(200, {"status": "ok", "session_id": sid})
            return

        # 问答
        if path == "/api/chat":
            question = (data.get("question") or "").strip()
            provider = data.get("provider")
            session_id = (data.get("session_id") or "").strip()
            user_role = (data.get("user_role") or "").strip()  # 前端身份选择器传入
            if not question:
                self._send(400, error_payload("question 不能为空", code=400))
                return
            user_id = resolve_user(self.headers, self.client_address[0])
            ip = self.client_address[0]
            req_id = uuid.uuid4().hex[:12]

            # ① 限流：用户 + IP + 全局 三维度
            rl = get_ratelimiter()
            ok, retry = rl.check(f"user:{user_id}", settings.rl_user_per_min, 60)
            if not ok:
                self._send_429("单用户请求过于频繁，请稍后再试", req_id, retry)
                return
            ok, retry = rl.check(f"ip:{ip}", settings.rl_ip_per_min, 60)
            if not ok:
                self._send_429("当前 IP 请求过于频繁，请稍后再试", req_id, retry)
                return
            ok, retry = rl.check("global", settings.rl_global_per_min, 60)
            if not ok:
                self._send_429("服务繁忙，请稍后再试", req_id, retry)
                return

            # ② 输入护栏：prompt 注入检测（PII 仅标记，不阻断当事人描述案情）
            gr = get_guardrails()
            ok_in, reason, has_pii = gr.scan_input(question)
            if not ok_in:
                log.warning("[%s] 输入被护栏拦截 | reason=%s | user=%s", req_id, reason, user_id)
                audit("blocked_input", trace_id=req_id, user_id=user_id, ip=ip, reason=reason)
                self._send(400, error_payload("输入含不安全内容，已被拦截", req_id, 400), trace_id=req_id)
                return

            # ③ 会话归属校验 + 短期记忆加载
            store = get_session_store()
            if session_id:
                sess = store.get_session(session_id)
                if sess is None or (sess.get("owner") and sess.get("owner") != user_id):
                    self._send(404, error_payload("会话不存在", req_id, 404), trace_id=req_id)
                    return
                history = store.get_messages(session_id)
            else:
                # 兼容旧调用：无 session_id 则新建会话并把前端传来的 history 种入
                session_id = store.create_session(owner=user_id)
                history = data.get("history") or []
                if isinstance(history, list):
                    for m in history:
                        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "bot") and m.get("content"):
                            store.append_message(session_id, "user" if m["role"] == "user" else "assistant", m["content"])
            if not isinstance(history, list):
                history = []
            history = history[-12:]
            if provider and provider not in MODEL_REGISTRY:
                provider = None
            t0 = time.perf_counter()
            log.info("[%s] POST /api/chat | sid=%s | user=%s | provider=%s | 历史=%d | q=%r",
                     req_id, session_id, user_id, provider or settings.default_llm, len(history),
                     gr.redact(question)[:60])
            use_stream = bool(data.get("stream"))

            # ④ 语义缓存命中 → 短路返回（仅非流式；含 PII 的答案不缓存，避免泄露）
            if not use_stream:
                cached = get_semantic_cache().get(question)
                if cached is not None:
                    store.append_message(session_id, "user", question)
                    ans = gr.redact(cached)
                    store.append_message(session_id, "assistant", ans)
                    audit("chat", trace_id=req_id, user_id=user_id, ip=ip, session_id=session_id,
                          route="cache", model="semantic-cache", cached=True)
                    self._send(200, {
                        "answer": ans, "route": "cache", "sources": [], "self_rag_retries": 0,
                        "model": "semantic-cache", "latency_ms": 0,
                        "case_profile": get_profile_store().get(session_id),
                        "trace": {"steps": []}, "trace_id": req_id, "session_id": session_id, "cached": True,
                    }, trace_id=req_id)
                    return

            try:
                # 先落「用户问题」到会话存储，再生成答案
                store.append_message(session_id, "user", question)
                if use_stream:
                    self._stream_chat(session_id, question, provider, history, req_id, user_id, ip, has_pii, user_role=user_role or None)
                    return
                result = pipeline.query(question, provider, history=history, session_id=session_id, user_role=user_role or None, user_id=user_id)
                result["trace_id"] = result.get("trace_id", req_id)
                result["answer"] = gr.redact(result["answer"])  # ⑤ 输出护栏：PII 脱敏
                store.append_message(session_id, "assistant", result["answer"])
                result["session_id"] = session_id
                get_semantic_cache().put(question, result["answer"], has_pii)  # ④ 回填语义缓存
                # B 方案：写入用户级 question_log（跨 session/跨角色汇总）
                try:
                    get_user_memory().record_question(user_id, session_id, user_role or "", question)
                except Exception as _e:
                    log.warning("[%s] question_log 写入失败: %s", req_id, _e)
                audit("chat", trace_id=req_id, user_id=user_id, ip=ip, session_id=session_id,
                      route=result["route"], model=result["model"], latency_ms=result["latency_ms"])
                self._send(200, result, trace_id=result["trace_id"])
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                log.error("[%s] 处理异常 | code=500 | 耗时=%.0fms | %s", req_id, dt, e)
                self._send(500, error_payload(str(e), req_id, 500), trace_id=req_id)
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

        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        # /api/sessions/{id}
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "sessions":
            sid = unquote(parts[2])
            user_id = resolve_user(self.headers, self.client_address[0])
            sess = get_session_store().get_session(sid)
            if sess is None or (sess.get("owner") and sess.get("owner") != user_id):
                self._send(404, {"error": f"会话不存在：{sid}"})
                return
            get_session_store().delete_session(sid)
            self._send(200, {"status": "ok", "action": "delete", "session_id": sid})
            return
            self._send(200, {"status": "ok", "action": "delete", "session_id": sid})
            return
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
