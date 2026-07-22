"""大模型调用层（生成答案）。

统一用 OpenAI 兼容 HTTP 接口对接三家公有模型（httpx 直连，零额外 SDK 依赖）：
  - DeepSeek  (deepseek-chat)         —— 开发基准
  - 智谱 GLM  (glm-4-flash / plus)
  - 通义千问  (qwen-plus / max)

开启 mock 模式时返回基于检索上下文的模板答案，便于无 key 冒烟。
"""
from __future__ import annotations

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


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._http: httpx.Client | None = None

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=60.0)
        return self._http

    def chat(
        self,
        messages: list[dict],
        provider: ProviderName | None = None,
        temperature: float = 0.3,
    ) -> str:
        provider = provider or self.s.default_llm
        cfg = resolve_model(provider, self.s)

        if self.s.mock or not cfg["api_key"]:
            log.debug("LLM 走 mock 模式 | provider=%s model=%s", cfg["provider"], cfg["model"])
            return self._mock_reply(messages, cfg)

        url = f"{cfg['base_url']}/chat/completions"
        log.info("LLM 请求 | provider=%s model=%s | 消息数=%d", cfg["provider"], cfg["model"], len(messages))
        t0 = time.perf_counter()
        try:
            resp = self.http.post(
                url,
                headers={"Authorization": f"Bearer {cfg['api_key']}"},
                json={
                    "model": cfg["model"],
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            log.info("LLM 完成 | provider=%s 耗时=%.2fs 回复数=%d", cfg["provider"], time.perf_counter() - t0, len(content))
            return content
        except Exception as e:
            log.error("LLM 调用失败 | provider=%s: %s", cfg["provider"], e)
            raise

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
