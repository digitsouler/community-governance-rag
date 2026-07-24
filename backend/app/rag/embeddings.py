"""文本向量化。

默认走智谱 Embedding-3（OpenAI 兼容 HTTP 接口，使用 httpx 直连，零额外 SDK 依赖）。
无密钥时自动降级为确定性 mock 向量，保证管道在无网/无 key 环境下也能跑通冒烟测试。

生产级升级：安装 `qdrant-client`/本地 bge 后可在 config 中切换为本地向量化，见 README。
"""
from __future__ import annotations

import hashlib
import math
import time

import httpx

from app.config import Settings, get_settings
from app.log import get_logger

log = get_logger("rag.embeddings")


class EmbeddingClient:
    # 单批最大文本数：智谱 embedding 接口对批量 input 上限约 400，分批避免 400。
    # 取 64 在单次延迟与并发安全间取得平衡（相比原 16 减少 3/4 网络往返）。
    BATCH = 64

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.dim = self.s.vector_dim
        self._http: httpx.Client | None = None
        self.use_mock = self.s.mock or not self.s.zhipu_api_key

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=30.0)
        return self._http

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.use_mock:
            log.debug("Embedding 走 mock 模式 | 文本数=%d", len(texts))
            return [self._mock_vector(t) for t in texts]
        # 分批调用，规避接口批量上限
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            vecs.extend(self._embed_batch(texts[i:i + self.BATCH]))
        return vecs

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self.s.zhipu_base_url}/embeddings"
        log.info("Embedding 请求 | model=%s 文本数=%d", self.s.zhipu_embed_model, len(batch))
        t0 = time.perf_counter()
        try:
            resp = self.http.post(
                url,
                headers={"Authorization": f"Bearer {self.s.zhipu_api_key}"},
                json={"model": self.s.zhipu_embed_model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in ordered]
            log.info("Embedding 完成 | 耗时=%.2fs 维数=%d", time.perf_counter() - t0, len(vecs[0]) if vecs else 0)
            return vecs
        except Exception as e:
            log.error("Embedding 调用失败: %s", e)
            raise

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _mock_vector(self, text: str) -> list[float]:
        """基于字符哈希的确定性伪向量，仅用于本地冒烟，不可用于真实检索质量评估。"""
        vec = [0.0] * self.dim
        for i, ch in enumerate(text):
            h = int(hashlib.md5(f"{i}:{ch}".encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
