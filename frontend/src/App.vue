<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'

const messages = reactive([])
const input = ref('')
const loading = ref(false)
const models = ref([])
const currentModel = ref('deepseek')
const chatArea = ref(null)

const examples = [
  '楼上漏水导致我家天花板发霉怎么办',
  '邻居私装地锁占用公共车位，其他业主如何处理？',
  '家里反复发生肢体冲突，如何申请人身安全保护？'
]

onMounted(async () => {
  try {
    const r = await fetch('/api/models')
    const data = await r.json()
    models.value = data.models || []
    currentModel.value = data.default || 'deepseek'
  } catch (e) {
    console.warn('获取模型列表失败', e)
  }
})

async function send(text) {
  const question = (text ?? input.value).trim()
  if (!question || loading.value) return
  input.value = ''
  // 多轮对话：截取"提交本轮问题之前"的历史（成对 user/bot），供后端承接上下文
  const history = messages
    .filter(m => (m.role === 'user' || m.role === 'bot') && !m.loading && m.content)
    .slice(-8)
    .map(m => ({ role: m.role, content: m.content }))
  messages.push({ role: 'user', content: question })
  messages.push({ role: 'bot', content: '', loading: true })
  loading.value = true
  await nextTick()
  scrollToBottom()

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, provider: currentModel.value, history })
    })
    const data = await resp.json()
    const last = messages[messages.length - 1]
    last.loading = false
    last.content = data.answer || ''
    last.route = data.route
    last.sources = (data.sources || []).filter(s => s.score >= SOURCE_MIN_SCORE)
    last.retries = data.self_rag_retries
    last.model = data.model
    last.latency = data.latency_ms
    last.traceId = data.trace_id
    last.trace = data.trace || null
  } catch (e) {
    const last = messages[messages.length - 1]
    last.loading = false
    last.content = '请求失败：' + (e.message || e)
  } finally {
    loading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  if (chatArea.value) chatArea.value.scrollTop = chatArea.value.scrollHeight
}

// 新建会话：清空当前对话，开启一段全新上下文（历史随之重置）
function newChat() {
  if (loading.value) return
  messages.splice(0, messages.length)
  input.value = ''
}

const routeLabel = { retrieve: '检索回答', direct: '直接回答', clarify: '需澄清', out_of_domain: '超出范围' }

// 与后端 source_display_min_score 对齐：低于此相关度的命中视为噪音，不渲染来源卡片
const SOURCE_MIN_SCORE = 0.3

// ---------- 知识库后台 ----------
const tab = ref('chat')
const kbStats = reactive({ total: 0, published: 0, draft: 0, chunks: 0, categories: {} })
const kbDocs = reactive({ total: 0, items: [] })
const kbLoading = ref(false)
const kbPage = ref(1)
const kbPageSize = 15
const uploadFile = ref(null)
const dirInput = ref(null)
const importing = ref(false)
const publishingAll = ref(false)
// 勾选（发布选中草稿）
const selectedIds = ref([])
const selectedCount = computed(() => selectedIds.value.length)
// 全选状态：当前页所有项都被选中时显示"取消全选"
const allPageSelected = computed(() =>
  kbDocs.items.length > 0 && kbDocs.items.every(d => selectedIds.value.includes(d.id))
)

async function loadKbStats() {
  try {
    const r = await fetch('/api/kb/stats')
    const d = await r.json()
    Object.assign(kbStats, d.stats || {})
  } catch (e) { console.warn('获取知识库统计失败', e) }
}

async function loadKbDocs() {
  kbLoading.value = true
  try {
    const r = await fetch(`/api/kb/docs?page=${kbPage.value}&size=${kbPageSize}`)
    const d = await r.json()
    Object.assign(kbDocs, d)
  } catch (e) { console.warn('获取文档列表失败', e) }
  finally { kbLoading.value = false }
}

async function switchTab(t) {
  tab.value = t
  if (t === 'kb') { kbPage.value = 1; await loadKbStats(); await loadKbDocs() }
}

async function doUpload() {
  const f = uploadFile.value?.files?.[0]
  if (!f) return
  const b64 = await new Promise((res, rej) => {
    const fr = new FileReader()
    fr.onload = () => res((fr.result || '').split(',')[1])
    fr.onerror = rej
    fr.readAsDataURL(f)
  })
  try {
    const r = await fetch('/api/kb/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: f.name, content_base64: b64 })
    })
    const d = await r.json()
    if (!d.status) { alert('上传失败：' + (d.error || '')); return }
    await loadKbStats(); await loadKbDocs()
  } catch (e) { alert('上传失败：' + (e.message || e)) }
}

async function kbAction(id, action) {
  // 删除走 DELETE /api/kb/{id}（注意：后端不认 /api/kb/{id}/delete）
  if (action === 'delete') {
    try {
      const r = await fetch(`/api/kb/${id}`, { method: 'DELETE' })
      const d = await r.json()
      if (!d.status && r.status !== 200) { alert('删除失败：' + (d.error || '')); return }
      await loadKbStats(); await loadKbDocs()
    } catch (e) { alert('删除失败：' + (e.message || e)) }
    return
  }
  try {
    const r = await fetch(`/api/kb/${id}/${action}`, { method: 'POST' })
    const d = await r.json()
    if (!d.status && r.status !== 200) { alert('操作失败：' + (d.error || '')); return }
    await loadKbStats(); await loadKbDocs()
  } catch (e) { alert('操作失败：' + (e.message || e)) }
}

// 批量导入整个目录：前端选文件夹 → 逐文件读取 → 分批提交后端（后端直接发布进向量库）
async function doImportDir(e) {
  const picked = Array.from(e.target.files || [])
  if (!picked.length) return
  const supported = ['.md', '.txt', '.pdf', '.docx', '.doc']
  const files = picked.filter(f => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    return supported.includes(ext)
  })
  if (!files.length) { alert('该目录下没有支持的文件（MD / PDF / Word / TXT）'); e.target.value = ''; return }
  importing.value = true
  let added = 0
  const BATCH = 10
  try {
    for (let i = 0; i < files.length; i += BATCH) {
      const batch = files.slice(i, i + BATCH)
      const payload = await Promise.all(batch.map(f => new Promise((res, rej) => {
        const fr = new FileReader()
        fr.onload = () => res({ filename: f.name, content_base64: (fr.result || '').split(',')[1] })
        fr.onerror = rej
        fr.readAsDataURL(f)
      })))
      try {
        const r = await fetch('/api/kb/import-directory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ files: payload })
        })
        const d = await r.json()
        if (d.added) added += d.added
      } catch (err) { console.warn('导入批次失败', err) }
    }
    alert(`已导入 ${added} 个文件，已存为草稿（保存在后端 corpus/uploads）。\n请在列表中勾选需上架的文档，点击「发布选中草稿」真正进向量库。`);
    kbPage.value = 1
    await loadKbStats(); await loadKbDocs()
  } finally {
    importing.value = false
    e.target.value = ''
  }
}

// 全选 / 取消全选（选中列表所有行，跨分页拉取全部 id）
async function toggleSelectAll() {
  if (selectedIds.value.length > 0) { selectedIds.value = []; return }
  try {
    const r = await fetch('/api/kb/docs?size=10000')
    const d = await r.json()
    selectedIds.value = (d.items || []).map(x => x.id)
  } catch (e) { console.warn('拉取全量文档失败', e) }
}

// 发布选中的草稿进向量库
async function publishSelected() {
  if (!selectedIds.value.length || publishingAll.value) return
  publishingAll.value = true
  try {
    const r = await fetch('/api/kb/publish-selected', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: selectedIds.value })
    })
    const d = await r.json()
    if (d.published > 0) alert(`已发布 ${d.published} 篇选中草稿（${d.chunks} 个分块），现在可在「对话」页检索。`)
    else alert('没有可发布的选中项。')
    selectedIds.value = []
    await loadKbStats(); await loadKbDocs()
  } catch (e) { alert('发布失败：' + (e.message || e)) }
  finally { publishingAll.value = false }
}

// 批量删除选中（本地文件 + 向量库一起删）
const deleting = ref(false)
async function deleteSelected() {
  if (!selectedIds.value.length || deleting.value) return
  const n = selectedIds.value.length
  if (!confirm(`确定要删除选中的 ${n} 个文档吗？\n这将同时删除本地文件和向量库数据，不可恢复。`)) return
  deleting.value = true
  try {
    const r = await fetch('/api/kb/delete-selected', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: selectedIds.value })
    })
    const d = await r.json()
    alert(`已删除 ${d.deleted || 0} 个文档（本地文件+向量库已清除）${d.failed ? `，${d.failed} 个失败` : ''}`)
    selectedIds.value = []
    await loadKbStats(); await loadKbDocs()
  } catch (e) { alert('批量删除失败：' + (e.message || e)) }
  finally { deleting.value = false }
}

// 详情预览：从后端拉取正文弹窗显示
const detailDoc = ref(null)
const detailLoading = ref(false)
const detailText = ref('')
const detailTooLarge = ref(false)
async function openDetail(id) {
  detailDoc.value = kbDocs.items.find(d => d.id === id) || { id, title: id }
  detailLoading.value = true
  detailText.value = ''
  detailTooLarge.value = false
  try {
    const r = await fetch('/api/kb/' + encodeURIComponent(id) + '/content')
    const d = await r.json()
    if (d.too_large) detailTooLarge.value = true
    else detailText.value = d.text || ''
  } catch (e) { detailText.value = '读取失败：' + (e.message || e) }
  finally { detailLoading.value = false }
}
function closeDetail() {
  detailDoc.value = null
  detailText.value = ''
  detailTooLarge.value = false
}

function kbGoPage(delta) {
  const totalP = Math.ceil(kbDocs.total / kbPageSize) || 1
  const p = kbPage.value + delta
  if (p >= 1 && p <= totalP) { kbPage.value = p; loadKbDocs() }
}
</script>

<template>
  <div class="app">
    <header class="app-header">
      <h1>🤝 社区矛盾调解 RAG 助手 <span class="badge">Agentic RAG</span></h1>
      <div class="header-right">
        <nav class="tabs">
          <button :class="['tab', { active: tab === 'chat' }]" @click="switchTab('chat')">对话</button>
          <button :class="['tab', { active: tab === 'kb' }]" @click="switchTab('kb')">知识库</button>
        </nav>
        <select class="model-select" v-model="currentModel" v-if="tab === 'chat'">
          <option v-for="m in models" :key="m.provider" :value="m.provider">
            {{ m.label }}（{{ m.model }}）{{ m.available ? '' : '· 未配置key' }}
          </option>
        </select>
        <button class="new-chat" v-if="tab === 'chat'" :disabled="!messages.length || loading" @click="newChat">＋ 新建会话</button>
      </div>
    </header>

    <div v-if="tab === 'chat'" class="chat-area" ref="chatArea">
      <div v-if="messages.length === 0" class="empty">
        <h2>你好，我是社区矛盾调解助理</h2>
        <p>描述你遇到的邻里 / 物业 / 家庭矛盾，我会结合知识库给出处置建议、相关法条与调解步骤，并标注依据来源。</p>
        <div class="chips">
          <span class="chip" v-for="ex in examples" :key="ex" @click="send(ex)">{{ ex }}</span>
        </div>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <div class="avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
        <div class="bubble">
          <div v-if="m.role === 'bot' && m.route" class="route-tag">
            {{ routeLabel[m.route] || m.route }} · 模型 {{ m.model }}
            <span v-if="m.retries"> · 自纠错重试 {{ m.retries }} 次</span>
          </div>
          <div v-if="m.loading" class="typing">正在检索知识库并生成…</div>
          <div v-else>{{ m.content }}</div>

          <div v-if="m.sources && m.sources.length" class="sources">
            <div class="source-card" v-for="s in m.sources" :key="s.id">
              <div class="s-title">📌 {{ s.title }}</div>
              <div class="s-meta">{{ s.category }} · 相关度 {{ s.score }}</div>
              <div class="s-content">{{ s.content }}</div>
              <div class="s-law" v-if="s.legal_basis">⚖️ {{ s.legal_basis }}</div>
            </div>
          </div>

          <details v-if="m.trace" class="trace" :open="m.route === 'retrieve'">
            <summary>
              🔍 检索链路
              <span v-if="m.latency" class="trace-latency">总耗时 {{ m.latency }}ms</span>
              <span v-if="m.traceId" class="trace-id">trace:{{ m.traceId }}</span>
            </summary>
            <ol class="trace-steps">
              <li v-for="(s, k) in m.trace.steps" :key="k" class="trace-step">
                <span class="ts-stage">{{ s.stage }}</span>
                <span class="ts-ms">{{ s.ms != null ? s.ms + 'ms' : '–' }}</span>
                <span class="ts-detail">{{ s.detail }}</span>
              </li>
            </ol>
          </details>
        </div>
      </div>
    </div>

    <div v-if="tab === 'chat'" class="input-bar">
      <textarea
        v-model="input"
        rows="1"
        placeholder="描述矛盾事实，例如：一楼私装地锁占用公共车位…"
        @keydown.enter.exact.prevent="send()"
      ></textarea>
      <button class="send-btn" @click="send()" :disabled="loading">发送</button>
    </div>

    <!-- 知识库后台 -->
    <div v-if="tab === 'kb'" class="kb-area">
      <div class="kb-stats">
        <div class="stat-card"><div class="num">{{ kbStats.total }}</div><div class="lbl">文档总数</div></div>
        <div class="stat-card ok"><div class="num">{{ kbStats.published }}</div><div class="lbl">已发布</div></div>
        <div class="stat-card draft"><div class="num">{{ kbStats.draft }}</div><div class="lbl">草稿</div></div>
        <div class="stat-card"><div class="num">{{ kbStats.chunks }}</div><div class="lbl">知识分块</div></div>
      </div>

      <div class="kb-toolbar">
        <label class="upload-btn" :class="{ disabled: importing }">
          上传文档（MD/PDF/Word/TXT）
          <input ref="uploadFile" type="file" accept=".md,.txt,.pdf,.docx" @change="doUpload" :disabled="importing" hidden />
        </label>
        <label class="upload-btn alt" :class="{ disabled: importing }">
          {{ importing ? '导入中…' : '导入整个目录（存为草稿）' }}
          <input ref="dirInput" type="file" webkitdirectory directory multiple @change="doImportDir" :disabled="importing" hidden />
        </label>
        <button class="upload-btn ghost" :disabled="kbDocs.total === 0" @click="toggleSelectAll">
          {{ allPageSelected ? '取消全选' : '全选' }}
        </button>
        <button class="upload-btn ghost primary" :disabled="selectedCount === 0 || publishingAll" @click="publishSelected">
          {{ publishingAll ? '发布中…' : `发布选中草稿 (${selectedCount})` }}
        </button>
        <button class="upload-btn ghost danger-outline" :disabled="selectedCount === 0 || deleting" @click="deleteSelected">
          {{ deleting ? '删除中…' : `批量删除 (${selectedCount})` }}
        </button>
        <span class="hint">勾选文档后可批量发布或删除；发布仅对草稿生效，删除同时清除本地文件与向量库。</span>
      </div>

      <div v-if="kbLoading" class="kb-loading">加载中…</div>
      <table v-else class="kb-table">
        <thead>
          <tr>
            <th class="col-check"><input type="checkbox" :checked="allPageSelected" @change="toggleSelectAll" title="全选/取消全选"></th>
            <th>标题</th><th>类别</th><th>状态</th><th>分块</th><th>来源</th><th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in kbDocs.items" :key="d.id">
            <td class="col-check"><input type="checkbox" :value="d.id" v-model="selectedIds"></td>
            <td class="td-title">{{ d.title }}</td>
            <td>{{ d.category }}</td>
            <td>
              <span :class="['pill', d.status === 'published' ? 'ok' : 'draft']">
                {{ d.status === 'published' ? '已发布' : '草稿' }}
              </span>
            </td>
            <td>{{ d.chunk_count }}</td>
            <td class="td-src">{{ d.source }}</td>
            <td class="td-ops">
              <button class="op" @click="openDetail(d.id)">详情</button>
              <button v-if="d.status !== 'published'" class="op ok" @click="kbAction(d.id, 'publish')">发布</button>
              <button v-else class="op" @click="kbAction(d.id, 'unpublish')">下架</button>
              <button class="op danger" @click="kbAction(d.id, 'delete')">删除</button>
            </td>
          </tr>
          <tr v-if="!kbDocs.items.length"><td colspan="7" class="empty-row">暂无文档</td></tr>
        </tbody>
      </table>

      <!-- 分页 -->
      <div class="kb-pagination" v-if="kbDocs.total > kbPageSize">
        <span class="page-info">第 {{ kbPage }} / {{ Math.ceil(kbDocs.total / kbPageSize) }} 页，共 {{ kbDocs.total }} 条</span>
        <button class="op" :disabled="kbPage <= 1" @click="kbGoPage(-1)">上一页</button>
        <button class="op" :disabled="kbPage >= Math.ceil(kbDocs.total / kbPageSize)" @click="kbGoPage(1)">下一页</button>
      </div>

      <!-- 详情预览弹窗 -->
      <div v-if="detailDoc" class="modal-mask" @click.self="closeDetail">
        <div class="modal">
          <div class="modal-head">
            <div class="modal-titles">
              <div class="modal-title">{{ detailDoc.title }}</div>
              <div class="modal-sub">{{ detailDoc.category }} · {{ detailDoc.status === 'published' ? '已发布' : '草稿' }} · {{ detailDoc.source }}</div>
            </div>
            <button class="modal-close" @click="closeDetail">✕</button>
          </div>
          <div class="modal-body">
            <div v-if="detailLoading" class="kb-loading">加载中…</div>
            <div v-else-if="detailTooLarge" class="modal-note">文件较大（&gt;2MB 或正文超过 2 万字），无法在此预览，请到后端 <code>corpus/uploads</code> 目录用本地软件打开。</div>
            <pre v-else class="modal-text">{{ detailText }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.trace {
  margin-top: 10px;
  border: 1px solid #e3e8ef;
  border-radius: 10px;
  background: #f8fafc;
  padding: 6px 10px;
  font-size: 12px;
}
.trace > summary {
  cursor: pointer;
  font-weight: 600;
  color: #475569;
  display: flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}
.trace-latency {
  margin-left: auto;
  color: #2563eb;
  font-weight: 600;
}
.trace-id {
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 400;
}
.trace-steps {
  margin: 8px 0 2px;
  padding-left: 4px;
  list-style: none;
}
.trace-step {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
  border-bottom: 1px dashed #eef2f7;
}
.trace-step:last-child { border-bottom: none; }
.ts-stage {
  flex: 0 0 96px;
  color: #0f172a;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600;
}
.ts-ms {
  flex: 0 0 64px;
  color: #2563eb;
  text-align: right;
}
.ts-detail {
  flex: 1;
  color: #64748b;
  word-break: break-all;
}

/* ---------- 顶部 Tab 切换 ---------- */
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.tabs {
  display: inline-flex;
  background: #eef2f7;
  border-radius: 10px;
  padding: 3px;
}
.tab {
  border: none;
  background: transparent;
  padding: 6px 14px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  color: #475569;
}
.tab.active {
  background: #fff;
  color: #0f172a;
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(0,0,0,.1);
}
.new-chat {
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #0f172a;
  padding: 6px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
}
.new-chat:hover:not(:disabled) { background: #f1f5f9; }
.new-chat:disabled { opacity: 0.45; cursor: not-allowed; }

/* ---------- 知识库后台 ---------- */
.kb-area {
  flex: 1;
  overflow-y: auto;
  padding: 18px 22px;
}
.kb-stats {
  display: flex;
  gap: 14px;
  margin-bottom: 16px;
}
.stat-card {
  flex: 1;
  background: #fff;
  border: 1px solid #e3e8ef;
  border-radius: 12px;
  padding: 14px 16px;
  text-align: center;
}
.stat-card .num { font-size: 26px; font-weight: 700; color: #0f172a; }
.stat-card .lbl { font-size: 12px; color: #64748b; margin-top: 2px; }
.stat-card.ok .num { color: #16a34a; }
.stat-card.draft .num { color: #d97706; }

.kb-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.upload-btn {
  background: #2563eb;
  color: #fff;
  padding: 8px 14px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}
.upload-btn:hover { background: #1d4ed8; }
.upload-btn.alt { background: #0f766e; }
.upload-btn.alt:hover { background: #115e59; }
.upload-btn.ghost { background: #fff; color: #2563eb; border: 1px solid #2563eb; }
.upload-btn.ghost:hover:not(:disabled) { background: #eff6ff; }
.upload-btn.disabled, .upload-btn:disabled { opacity: .55; cursor: not-allowed; }
.upload-btn:disabled:hover { background: inherit; }
.hint { font-size: 12px; color: #94a3b8; }

.kb-table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border: 1px solid #e3e8ef;
  border-radius: 12px;
  overflow: hidden;
  font-size: 13px;
}
.kb-table th, .kb-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #eef2f7;
}
.kb-table th { background: #f8fafc; color: #475569; font-weight: 600; }
.kb-table tr:last-child td { border-bottom: none; }
.td-title { font-weight: 600; color: #0f172a; max-width: 280px; }
.td-src { color: #64748b; max-width: 220px; }
.empty-row { text-align: center; color: #94a3b8; padding: 24px; }

.pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
}
.pill.ok { background: #dcfce7; color: #16a34a; }
.pill.draft { background: #fef3c7; color: #d97706; }

.td-ops { white-space: nowrap; }
.op {
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #334155;
  padding: 4px 10px;
  border-radius: 7px;
  cursor: pointer;
  font-size: 12px;
  margin-right: 6px;
}
.op:hover { background: #f1f5f9; }
.op.ok { border-color: #16a34a; color: #16a34a; }
.op.danger { border-color: #ef4444; color: #ef4444; }
.kb-loading { color: #94a3b8; padding: 24px; text-align: center; }

.kb-pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 14px;
  padding: 10px 0;
}
.page-info { font-size: 13px; color: #64748b; }
.kb-pagination .op[disabled] { opacity: 0.4; cursor: not-allowed; }

.upload-btn.primary { background: #16a34a; border-color: #16a34a; color: #fff; }
.upload-btn.primary:hover:not(:disabled) { background: #15803d; }
.upload-btn.danger-outline { background: #fff; color: #ef4444; border: 1px solid #ef4444; }
.upload-btn.danger-outline:hover:not(:disabled) { background: #fef2f2; }

.col-check { width: 38px; text-align: center; }
.col-check input { width: 16px; height: 16px; cursor: pointer; }

/* ---------- 详情弹窗 ---------- */
.modal-mask {
  position: fixed; inset: 0;
  background: rgba(15, 23, 42, .45);
  display: flex; align-items: center; justify-content: center;
  z-index: 50; padding: 24px;
}
.modal {
  background: #fff; border-radius: 14px;
  width: min(820px, 92vw); max-height: 86vh;
  display: flex; flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,.25);
  overflow: hidden;
}
.modal-head {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 16px 18px; border-bottom: 1px solid #eef2f7;
}
.modal-titles { flex: 1; min-width: 0; }
.modal-title { font-size: 16px; font-weight: 700; color: #0f172a; word-break: break-all; }
.modal-sub { font-size: 12px; color: #64748b; margin-top: 4px; word-break: break-all; }
.modal-close {
  border: none; background: #f1f5f9; color: #475569;
  width: 30px; height: 30px; border-radius: 8px; cursor: pointer; font-size: 15px;
}
.modal-close:hover { background: #e2e8f0; }
.modal-body { padding: 16px 18px; overflow: auto; }
.modal-text {
  margin: 0; white-space: pre-wrap; word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; line-height: 1.7; color: #1e293b;
  background: #f8fafc; border: 1px solid #eef2f7; border-radius: 10px; padding: 14px;
  max-height: 60vh; overflow: auto;
}
.modal-note { color: #d97706; background: #fffbeb; border: 1px solid #fde68a; padding: 12px 14px; border-radius: 10px; font-size: 13px; }
.modal-note code { background: #fef3c7; padding: 1px 5px; border-radius: 4px; }
</style>
