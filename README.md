# 社区矛盾调解 RAG 助手

面向基层社区治理的 **Agentic RAG** 应用：把矛盾调解知识（邻里噪音、漏水、停车、物业、赡养、家暴等）接入检索增强生成，让调解员和居民在描述纠纷时，快速获得**有依据、可追溯、不编造**的处置建议、相关法条与调解步骤。

> 场景痛点：社区矛盾类型多、法条分散、新人上手慢。传统"搜文档"效率低，通用聊天机器人又容易凭空编造法条。本项目用 RAG + 智能路由 + 自纠错，让答案**每一句都来自知识库、每一条都能溯源**。

## 核心能力

1. **Supervisor 智能路由**：先判断问题类型——问候/能力问答直接回答，信息过少则引导补充，只有真正需要时才走检索，省 token 也更准。
2. **Self-RAG 自纠错**：检索质量不达标时自动改写查询重试；多次仍无相关依据则**诚实告知"知识库暂无依据"**，绝不编造法条。
3. **多模型可切换横评**：同一套 RAG 管道，可在 DeepSeek / 智谱 GLM / 通义千问之间一键切换，为后续做 faithfulness、延迟、成本的横向对比打底。
4. **引用溯源**：每条答复附带来源卡片（标题、类型、相关度、内容摘要、法条原文），便于人工复核。

## 架构

```
┌────────────┐    /api/chat    ┌───────────────────────────┐
│  Vue 3 前端 │ ─────────────▶ │  FastAPI / http.server 服务 │
│ (Vite)     │ ◀───────────── │                           │
└────────────┘    JSON 答复    │  ┌─────────────────────┐  │
                              │  │  RAG Pipeline        │  │
                              │  │  ├ Supervisor 路由   │  │
                              │  │  ├ 向量检索 + 重排   │  │
                              │  │  ├ Self-RAG 自纠错  │  │
                              │  │  └ 大模型生成(溯源) │  │
                              │  └─────────────────────┘  │
                              │    │            │         │
                              │  ┌─▼────────┐ ┌─▼──────┐ │
                              │  │ 向量库   │ │ LLM/   │ │
                              │  │(内存/Qdrant)│ Embedding│ │
                              │  └──────────┘ └────────┘ │
                              └───────────────────────────┘
```

- **编排**：自研轻量 Agentic 控制器（Supervisor + Self-RAG 循环）
- **向量库**：默认纯内存（零依赖）；配置 `QDRANT_URL` 可切换 Qdrant 服务端
- **Embedding**：智谱 Embedding-3（OpenAI 兼容接口）
- **生成模型**：DeepSeek（基准）/ 智谱 GLM / 通义千问（均 OpenAI 兼容）
- **评测**：RAGAS（规划中，见下文）

## 快速开始

### 1. 后端

```bash
cd backend
cp .env.example .env          # 填入你的模型 API Key（至少一项；默认 DeepSeek）
pip install -r requirements.txt   # 或仅 pip install httpx
python -m app.main             # 启动后访问 http://localhost:8000/api/health
```

不填任何 Key 也能跑：自动进入 **mock 模式**，管道逻辑全链路可验证（答案由模板生成，仅用于冒烟）。

### 2. 前端

```bash
cd frontend
npm install
npm run dev                   # 访问 http://localhost:5173
```

前端开发服务器已配置 `/api` 代理到后端 `:8000`，开箱即用。

## 目录结构

```
.
├── backend/
│   ├── app/
│   │   ├── config.py         # 配置（模型/向量库/检索参数，全部走环境变量）
│   │   ├── main.py           # 零依赖 HTTP 服务入口（生产可换 FastAPI 版）
│   │   ├── rag/
│   │   │   ├── embeddings.py # 向量化（智谱 API，含 mock 降级）
│   │   │   ├── llm.py        # 大模型调用（DeepSeek/智谱/Qwen 统一接口）
│   │   │   ├── vectorstore.py# 向量库（内存 / 可选 Qdrant）
│   │   │   ├── rerank.py     # 重排序（向量+词面混合，可升级 bge）
│   │   │   └── pipeline.py   # 核心：Supervisor 路由 + Self-RAG 自纠错
│   │   └── data/
│   │       ├── mediation_cases.json  # 矛盾调解样例知识库
│   │       └── ingest.py             # 入库脚本
│   └── tests/smoke.py        # 冒烟测试
└── frontend/                 # Vue 3 + Vite 聊天界面
```

## 评测与横向对比（规划）

- 用 RAGAS 对 faithfulness / answer_relevancy / context_precision 做自动化评测
- 同一测试集下对比 DeepSeek / 智谱 / Qwen 在**同一 RAG 管道**中的表现，产出 Benchmark 表

## 调试与可观测性

每个请求都附带可追踪信息，方便定位问题与优化：

- **结构化日志**：后端使用标准库 `logging`，控制台 + `backend/logs/app.log`（滚动 5MB×3）双输出，按 `LOG_LEVEL`（默认 `info`，可设 `debug`）分级。
- **请求级 Trace**：每次 `/api/chat` 返回 `trace_id` 与分阶段耗时 `trace.steps`（路由 / 向量检索 / 重排 / 大模型生成），并写入响应头 `X-Trace-Id`。前端在每条回答下展开「🔍 检索链路」面板即可看到完整时间线。
- **诚实可观测**：路由判定、检索最佳相关度、Self-RAG 重试次数、来源召回情况均会记入日志与接口返回，便于复现与评测。

```bash
tail -f backend/logs/app.log     # 实时跟踪后端运行
```

> 生产环境如需更专业的 LLM 调用追踪（token、成本、对话回放），可接入 Langfuse / LangSmith / OpenTelemetry，本项目已在 `llm.py` 与 `pipeline.py` 预留日志埋点，接入成本低。

## 说明

当前仓库可在**零额外依赖**下运行（仅标准库 + `httpx`）。如希望使用 Qdrant 向量库或 FastAPI 生产栈，按 `requirements.txt` 安装对应包并在 `config` / 入口处切换即可，核心逻辑无需改动。
