"""全局配置：从环境变量 / .env 读取，所有密钥均走配置，不在代码里硬编码。"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["deepseek", "zhipu", "qwen"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ---- 默认模型（开发基准 = DeepSeek）----
    default_llm: ProviderName = "deepseek"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 智谱 GLM
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"
    zhipu_embed_model: str = "embedding-3"

    # 通义千问 Qwen
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    # ---- 向量库 ----
    # 留空则用 Qdrant 内存模式（无需起服务），填 url 则用远程 Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "community_mediation"
    vector_dim: int = 2048  # 智谱 embedding-3 维度

    # ---- 检索参数 ----
    top_k: int = 8
    rerank_top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 80

    # ---- 自纠错（Self-RAG）----
    max_retrieve_retries: int = 2
    relevance_threshold: float = 0.15  # 低于此分数判定检索不相关，触发改写重试

    # ---- Mock 模式（无密钥也能跑通管道，仅用于本地冒烟）----
    mock: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


# 各供应商的模型端点元信息，供接口层动态切换
MODEL_REGISTRY: dict[ProviderName, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url_env": "deepseek_base_url",
        "model_env": "deepseek_model",
        "key_env": "deepseek_api_key",
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url_env": "zhipu_base_url",
        "model_env": "zhipu_model",
        "key_env": "zhipu_api_key",
    },
    "qwen": {
        "label": "通义千问",
        "base_url_env": "qwen_base_url",
        "model_env": "qwen_model",
        "key_env": "qwen_api_key",
    },
}
