"""大模型调用层（生成答案）。

统一用 OpenAI 兼容 HTTP 接口对接三家公有模型（httpx 直连，零额外 SDK 依赖）：
  - DeepSeek  (deepseek-chat)         —— 开发基准
  - 智谱 GLM  (glm-4-flash / plus)
  - 通义千问  (qwen-plus / max)

开启 mock 模式时返回基于检索上下文的模板答案，便于无 key 冒烟。
"""
from __future__ import annotations

import json
import time

import httpx

from app.config import MODEL_REGISTRY, ProviderName, Settings, get_settings
from app.log import get_logger

log = get_logger("rag.llm")


def resolve_model(provider: ProviderName, settings: Settings):
    meta = MODEL_REGISTRY[provider]
    return {
        "provider": provider,
        "label": meta["label"],
        "base_url": getattr(settings, meta["base_url_env"]),
        "api_key": getattr(settings, meta["key_env"]),
        "model": getattr(settings, meta["model_env"]),
    }


class _Breaker:
    """单 provider 熔断器：窗口内失败率超阈值即熔断，避免线程堆积在故障模型上。"""

    def __init__(self, name: str, threshold: float, min_calls: int, open_seconds: int):
        self.name = name
        self.threshold = threshold
        self.min_calls = min_calls
        self.open_seconds = open_seconds
        self.calls: list[bool] = []
        self.open_until = 0.0

    def allow(self, now: float) -> bool:
        return now >= self.open_until

    def record(self, ok: bool, now: float) -> None:
        self.calls.append(ok)
        if len(self.calls) > self.min_calls * 2:
            self.calls = self.calls[-self.min_calls * 2:]
        if len(self.calls) >= self.min_calls:
            fail = sum(1 for c in self.calls if not c)
            if fail / len(self.calls) >= self.threshold:
                self.open_until = now + self.open_seconds
                log.warning("熔断开启 %s，%ds 内不再直连（失败率超阈值）", self.name, self.open_seconds)


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._http: httpx.Client | None = None
        self._breakers = {
            p: _Breaker(p, self.s.circuit_error_rate, self.s.circuit_min_calls, self.s.circuit_open_seconds)
            for p in MODEL_REGISTRY
        }

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.s.llm_timeout)
        return self._http

    def _order(self, provider: ProviderName | None) -> list[ProviderName]:
        """主模型优先，其后接配置的降级链。"""
        provider = provider or self.s.default_llm
        others = [p for p in self.s.fallback_providers if p in MODEL_REGISTRY and p != provider]
        return [provider, *others]

    def _chat_once(self, provider: ProviderName, messages: list[dict], temperature: float) -> str:
        cfg = resolve_model(provider, self.s)
        url = f"{cfg['base_url']}/chat/completions"
        log.info("LLM 请求 | provider=%s model=%s | 消息数=%d", cfg["provider"], cfg["model"], len(messages))
        t0 = time.perf_counter()
        resp = self.http.post(
            url,
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            json={"model": cfg["model"], "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        log.info("LLM 完成 | provider=%s 耗时=%.2fs 回复数=%d", cfg["provider"], time.perf_counter() - t0, len(content))
        return content

    def chat(self, messages: list[dict], provider: ProviderName | None = None, temperature: float = 0.3) -> str:
        provider = provider or self.s.default_llm
        cfg = resolve_model(provider, self.s)
        if self.s.mock or not cfg["api_key"]:
            return self._mock_reply(messages, cfg)
        errors = []
        for p in self._order(provider):
            br = self._breakers[p]
            now = time.time()
            if not br.allow(now):
                errors.append(f"{p}:circuit_open")
                continue
            try:
                content = self._chat_once(p, messages, temperature)
                br.record(True, now)
                return content
            except Exception as e:  # noqa: BLE001
                br.record(False, now)
                errors.append(f"{p}:{type(e).__name__}")
                # 瞬时失败重试一次（指数退避），仍失败则走降级链
                if self.s.llm_max_retries > 0:
                    try:
                        time.sleep(0.5)
                        content = self._chat_once(p, messages, temperature)
                        br.record(True, now)
                        return content
                    except Exception as e2:  # noqa: BLE001
                        br.record(False, now)
                        errors.append(f"{p}-retry:{type(e2).__name__}")
        raise RuntimeError("所有模型均不可用: " + "; ".join(errors))

    def _stream_once(self, provider: ProviderName, messages: list[dict], temperature: float):
        cfg = resolve_model(provider, self.s)
        url = f"{cfg['base_url']}/chat/completions"
        log.info("LLM 流式请求 | provider=%s model=%s | 消息数=%d", cfg["provider"], cfg["model"], len(messages))
        with self.http.stream(
            "POST", url,
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            json={"model": cfg["model"], "messages": messages, "temperature": temperature, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:  # noqa: BLE001
                        continue

    def stream_chat(self, messages: list[dict], provider: ProviderName | None = None, temperature: float = 0.3):
        """流式生成（带降级链 + 熔断）：主模型失败自动切换到备用模型。"""
        provider = provider or self.s.default_llm
        cfg = resolve_model(provider, self.s)
        if self.s.mock or not cfg["api_key"]:
            yield self._mock_reply(messages, cfg)
            return
        errors = []
        for p in self._order(provider):
            br = self._breakers[p]
            if not br.allow(time.time()):
                errors.append(f"{p}:circuit_open")
                continue
            try:
                yield from self._stream_once(p, messages, temperature)
                br.record(True, time.time())
                return
            except Exception as e:  # noqa: BLE001
                br.record(False, time.time())
                log.warning("LLM 流式失败，尝试降级链 | %s: %s", p, e)
                errors.append(f"{p}:{type(e).__name__}")
        raise RuntimeError("所有模型流式均不可用: " + "; ".join(errors))

    def _mock_reply(self, messages: list[dict], cfg: dict) -> str:
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break
        tag = "【参考依据】"
        if tag in last_user:
            context = last_user.split(tag, 1)[1]
            return (
                f"（Mock 模式 · {cfg['label']}）根据知识库检索到以下内容，"
                f"建议据此组织调解：\n{context[:400]}……\n"
                f"注：当前为本地冒烟响应，配置真实 API Key 后由大模型生成完整答复。"
            )
        return f"（Mock 模式 · {cfg['label']}）已收到您的消息，管道连通正常。"


def list_available_models(settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    out = []
    for name, meta in MODEL_REGISTRY.items():
        key = getattr(s, meta["key_env"])
        out.append(
            {
                "provider": name,
                "label": meta["label"],
                "model": getattr(s, meta["model_env"]),
                "available": bool(key) and not s.mock,
            }
        )
    return out
