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
- **评测**：RAGAS 四指标横评（DeepSeek / 智谱 / Qwen 同管道对比，已跑通，见 `backend/benchmark_report.md`）

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
│   │   ├── ingest/           # 入库管道：文件解析 → 切分 → 向量化 → 增量 upsert
│   │   │   ├── loaders.py    # PDF/Word/MD/TXT 加载（OCR 钩子，懒加载）
│   │   ├── kb/               # 知识库后台：文档生命周期（草稿→发布→下架→删除）
│   │   │   └── __init__.py   # KBManager：以 corpus/docs + corpus/uploads 文件为数据源
│   │   │   ├── split.py      # 中文友好切分（段落/句子 + 重叠）
│   │   │   ├── pipeline.py   # 编排 + 按 doc_id 增量去重
│   │   │   ├── gen_sample_corpus.py  # 生成 160 篇演示语料（写入 corpus/docs/）
│   │   │   └── __main__.py   # CLI：python -m app.ingest
│   │   ├── rag/
│   │   │   ├── embeddings.py # 向量化（智谱 API，含 mock 降级 + 自动分批）
│   │   │   ├── llm.py        # 大模型调用（DeepSeek/智谱/Qwen 统一接口）
│   │   │   ├── vectorstore.py# 向量库（内存 / 可选 Qdrant，含零依赖 REST 兜底）
│   │   │   ├── rerank.py     # 重排序（向量+词面混合，可升级 bge）
│   │   │   └── pipeline.py   # 核心：Supervisor 路由 + Self-RAG 自纠错（BM25 从向量库 payload 重建）
│   │   └── data/
│   │       ├── mediation_cases.json  # 48 篇种子案例（评测用；种子知识库已文件化为 corpus/docs/seeds/）
│   │       ├── kb_index.json         # 轻量索引：仅存元信息（id/状态/路径/分块数），不含正文
│   │       └── ingest.py             # 旧版入库脚本（保留兼容；新架构以文件为数据源）
│   ├── corpus/
│   │   ├── docs/            # 知识库语料目录（16 类演示语料 + seeds/ 48 篇种子，可替换为自有文档）
│   │   └── uploads/         # 用户上传 / 导入的文件（默认草稿，审核后发布）
│   └── tests/                # 冒烟 / 评测 / RAGAS 横评脚本
└── frontend/                 # Vue 3 + Vite 聊天界面
```

## 评测与横向对比

已用 RAGAS 在同一 RAG 管道下对 **DeepSeek / 智谱 GLM / 通义千问** 做四指标横评（faithfulness / answer_relevancy / context_precision / context_recall），报告见 [`backend/benchmark_report.md`](backend/benchmark_report.md)。

- 轻量启发式横评（要点覆盖 / 来源准确 / 拒答正确 / 延迟）：`python tests/heuristic_eval.py --provider all`
- RAGAS 权威四指标（本机跑，详见报告）：`PYTHONPATH=. python tests/run_ragas.py --provider all`

> 评测结论：三模型要点覆盖率均 100%、拒答均正确；质量差距很小，**DeepSeek 性价比最高**（相关度精度最高且延迟最低）。当前最大短板是 faithfulness（答案偶有超出知识库的表述），已在优化生成 prompt 与扩充语料（见下）。

## 调试与可观测性

每个请求都附带可追踪信息，方便定位问题与优化：

- **结构化日志**：后端使用标准库 `logging`，控制台 + `backend/logs/app.log`（滚动 5MB×3）双输出，按 `LOG_LEVEL`（默认 `info`，可设 `debug`）分级。
- **请求级 Trace**：每次 `/api/chat` 返回 `trace_id` 与分阶段耗时 `trace.steps`（路由 / 向量检索 / 重排 / 大模型生成），并写入响应头 `X-Trace-Id`。前端在每条回答下展开「🔍 检索链路」面板即可看到完整时间线。
- **诚实可观测**：路由判定、检索最佳相关度、Self-RAG 重试次数、来源召回情况均会记入日志与接口返回，便于复现与评测。

```bash
tail -f backend/logs/app.log     # 实时跟踪后端运行
```

> 生产环境如需更专业的 LLM 调用追踪（token、成本、对话回放），可接入 Langfuse / LangSmith / OpenTelemetry，本项目已在 `llm.py` 与 `pipeline.py` 预留日志埋点，接入成本低。

## 入库管道（生产数据层）

演示用的 48 篇样例是手写 JSON。生产环境需要把真实文档（政策 PDF、案例 Word、扫描件等）**持续入库**。`app/ingest/` 解决这件事，且与业务场景解耦——换场景只换语料目录，管道代码不用改。

### 1) 准备语料

```bash
cd backend
# 生成 160 篇演示语料（16 类矛盾，每类 10 篇）到 corpus/raw/
PYTHONPATH=. python -m app.ingest.gen_sample_corpus
```

你也可以直接把单位的 **PDF / Word / Markdown / 文本** 丢进 `corpus/raw/` 任意子目录，无需改代码。

### 2) 入库

```bash
# 增量入库：按 doc_id 去重，重跑只补新文档，不重复烧 embedding
PYTHONPATH=. python -m app.ingest

# 全量重建（先清空向量库与索引）
PYTHONPATH=. python -m app.ingest --reset

# 多租户隔离（预留，向量库按 payload.tenant_id 过滤即可）
PYTHONPATH=. python -m app.ingest --tenant-id acme
```

- 支持的格式：`.md / .txt / .json / .pdf / .docx`。PDF / Word 解析依赖 `pip install pypdf python-docx`；扫描件（图片型 PDF）另需 `paddleocr pdf2image` 或配置外部 OCR 服务。
- 入库后**重启后端服务**即生效：BM25 稀疏索引在服务启动时从统一检索源（种子 + 文件语料）自动重建，混合检索即可命中新文档。
- 服务端启动时会自动把 `corpus/raw/` 增量灌入，所以克隆仓库后直接 `python -m app.main` 也能用。

> **两条入库入口的区别**
> - `python -m app.ingest`：把 `corpus/raw/` 下的**文件文档**解析入库（写入 `ingested.jsonl`，若已配 `QDRANT_URL` 则同步 upsert）。用于日常「新增/更新语料」。
> - `python -m app.data.ingest`（或启动服务端）：加载 **48 条种子案例 + `ingested.jsonl` 全部文件语料**，一次性重建完整知识库并 upsert 到向量库。用于「首次全量灌库 / 换向量库后重建」。

### 3) 本地向量库（Qdrant，免费）

默认是纯内存向量库，重启需重新入库。需要**持久化**时，起一个本地 Qdrant（Windows 原生程序 / Docker 均可，完全免费、架构与云端同构）：

```bash
# 方式 A：Docker
docker compose up -d
# 方式 B：Windows 原生 Qdrant
#   下载 qdrant.exe 直接运行即可（默认暴露 :6333）

# 然后在 backend/.env 中设置（二者相同）：
QDRANT_URL=http://localhost:6333
```

配置后**无需改代码**，入库与检索自动走 Qdrant，重启不丢数据。几点说明：

- **无需安装 `qdrant-client`**：若环境没装该包，代码会自动回退到零依赖的 **REST 适配**（`vectorstore.QdrantRestStore`，基于 `httpx`），功能完全一致。
- **首次全量灌库**：设好 `QDRANT_URL` 后，跑一次完整知识库构建即可把 48 种子 + 全部文件语料写进 Qdrant：

  ```bash
  cd backend
  PYTHONPATH=. python -m app.data.ingest        # 加载 48 种子 + ingested.jsonl，全量 upsert
  # 等价于启动服务端时自动执行的那一步；此后检索都走 Qdrant
  ```

- 托管云（Qdrant Cloud / Zilliz）只在你要**公网多人在线访问**时才需要——本地演示不必花钱。

### 4) 多场景 / 多租户

- **多场景**：入库管道场景无关。新增场景只需把对应文档放入 `corpus/raw/<场景名>/` 子目录并重跑入库，零额外开发。
- **多租户（架构预留）**：入库 payload 已带 `tenant_id` 字段；向量库按 namespace 隔离即可支持多租户，当前仓库为单租户演示。

## 知识库后台（可视化运营）

除命令行入库外，项目内置一个**知识库后台**，可在浏览器里完成「上传 → 审核 → 发布 / 下架 / 删除」的完整生命周期管理，无需碰命令行。前端「对话 / 知识库」一键切换即可。

**能力**
- **统计看板**：文档总数 / 已发布 / 草稿 / 知识分块数，指标一目了然。
- **上传文档**：MD / TXT / PDF / Word 直接上传，落盘后先成为**草稿**，不进检索。
- **发布 / 下架 / 删除**：发布才嵌入向量库并进入检索；下架立即从向量库移除；删除连同源文件一并清理。所有操作实时同步 BM25 稀疏索引。
- **草稿态审核**：只有 `published` 的文档参与检索与回答，保证「先审后上」，避免错误资料直接暴露给用户。

**实现要点**
- 单一事实来源是 `backend/data/kb_manifest.jsonl`（文档级 manifest，记录元信息 / 分块 / 状态 / 来源路径）；已发布文档才进入向量库 + BM25。
- 每个分块 payload 携带 `doc_id`，支持按文档整体上下架（`vectorstore.delete_by_doc_id`），无需逐块操作。
- 首次启动 `KBManager.ensure()` 会自动把 **48 条种子案例 + `corpus/raw/` 全部文件语料** 迁移进 manifest 并置为已发布，克隆仓库即可用。

**接口**（零依赖 `http.server` 已内置，生产切换 FastAPI 接口一致）

```
GET  /api/kb/stats                  知识库统计
GET  /api/kb/docs?status=&page=     文档列表（支持按状态筛选）
POST /api/kb/upload                  上传文档（base64 JSON）-> 草稿
POST /api/kb/{id}/publish            发布（嵌入 + 入向量库 + 重建 BM25）
POST /api/kb/{id}/unpublish          下架（从向量库移除）
DELETE /api/kb/{id}                  删除（含物理文件）
```

## 说明

当前仓库可在**零额外依赖**下运行（仅标准库 + `httpx`）。如希望使用 Qdrant 向量库或 FastAPI 生产栈，按 `requirements.txt` 安装对应包并在 `config` / 入口处切换即可，核心逻辑无需改动。
