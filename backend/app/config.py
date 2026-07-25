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

    # ---- 短期记忆（会话存储）----
    # 留空则内存降级模式
    # 填 redis://host:port 启用 Redis 持久化会话
    redis_url: str = "redis://127.0.0.1:6379"

    # ---- 网关层：多用户隔离 / 限流 ----
    rate_limit_enabled: bool = True
    rl_user_per_min: int = 12        # 单用户每分钟请求上限（防单个账号狂刷）
    rl_ip_per_min: int = 40          # 单 IP 每分钟上限（防同 IP 多账号/匿名蹭）
    rl_global_per_min: int = 600      # 全局兜底（防总过载）
    auth_required: bool = False       # demo 不强制登录，前端自生成稳定 user_id 即可

    # ---- 网关层：弹性降级（LLM 超时 / 重试 / 熔断 / 降级链）----
    llm_timeout: float = 45.0         # 单次 LLM 调用硬超时（秒）
    llm_max_retries: int = 1          # 瞬时失败（429/5xx/超时）重试次数，指数退避
    circuit_error_rate: float = 0.5   # 熔断错误率阈值（窗口内失败占比）
    circuit_min_calls: int = 10       # 进入熔断统计的最小调用数
    circuit_open_seconds: int = 30    # 熔断持续时间
    fallback_providers: list[str] = ["zhipu", "qwen"]  # 主模型失败时的降级链

    # ---- 网关层：语义缓存（查询向量近邻命中短路）----
    semantic_cache_enabled: bool = True
    semantic_cache_sim: float = 0.92  # 余弦相似度命中阈值
    semantic_cache_size: int = 500    # 进程内缓存容量上限

    # ---- 网关层：护栏 / 审计 ----
    guardrails_enabled: bool = True   # 输入 prompt 注入检测 + 输出 PII 脱敏
    pii_redact_enabled: bool = True
    audit_enabled: bool = True        # append-only 审计日志（user/ip/session/命中）

    # ---- 检索参数 ----
    top_k: int = 8
    rerank_top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 80

    # ---- 自纠错（Self-RAG）----
    max_retrieve_retries: int = 2
    relevance_threshold: float = 0.15  # 低于此分数判定检索不相关，触发改写重试

    # ---- 混合检索（Hybrid Retrieval）----
    # 稠密向量召回 ∪ BM25 稀疏召回，经 RRF 融合后送 rerank 精排。
    # 关闭则退化为纯向量召回（与旧版一致）；开启可补专有词/法条条号召回。
    enable_hybrid: bool = True
    rrf_k: int = 60  # RRF 融合常数

    # ---- 来源展示门槛 ----
    # 检索命中的文档，相关度低于此值视为「噪音」不展示给用户
    # （0.3 是经验值：真实语义匹配普遍 0.5+，<0.3 基本是随机凑数）
    source_display_min_score: float = 0.3

    # ---- Mock 模式（无密钥也能跑通管道，仅用于本地冒烟）----
    mock: bool = False

    # ---- 查询分解（复杂纠纷分步检索合并）----
    # 开启后，对复杂多子问题（长句 / 多问号 / 并列诉求）自动拆成多个子问题，
    # 分别检索再合并重排；简单问题不受影响（走原有单查询路径）。
    enable_decomposition: bool = True
    max_sub_queries: int = 4


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
