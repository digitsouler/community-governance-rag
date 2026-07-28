<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'

const messages = reactive([])
const input = ref('')
const loading = ref(false)
const models = ref([])
const currentModel = ref('deepseek')
const chatArea = ref(null)

// 短期记忆：会话列表与当前会话 id（后端 Redis/内存存储，刷新/新标签不丢）
const sessionId = ref('')
const sessions = ref([])

// 长期记忆：当前会话沉淀的结构化案件档案（可视化给用户看）
const caseProfile = ref(null)
const profileOpen = ref(false)

// 多用户隔离：前端自生成稳定 user_id（demo 级身份，非登录鉴权），随请求带上供后端隔离/限流
const USER_ID_KEY = 'cg_user_id'
let userId = localStorage.getItem(USER_ID_KEY)
if (!userId) {
  userId = 'u_' + (crypto.randomUUID ? crypto.randomUUID().slice(0, 12) : Math.random().toString(36).slice(2, 14))
  localStorage.setItem(USER_ID_KEY, userId)
}
const AUTH = { 'X-User-Id': userId }

// 身份选择器：首次对话前引导用户选择角色，后续每条请求都带上 user_role
const ROLE_OPTIONS = [
  { value: 'resident', label: '居民 / 业主', desc: '遇到矛盾的一方，想了解怎么办' },
  { value: 'property', label: '物业人员', desc: '物业服务/管家/工程/安保视角' },
  { value: 'mediator', label: '调解员 / 居委会', desc: '社区调解/居委会工作人员' },
  { value: 'grid_worker', label: '网格员 / 社工', desc: '一线走访/社工/网格员视角' },
]
const userRole = ref('')
const showRolePicker = computed(() => !userRole.value && messages.length === 0)
function updateProfile(p) {
  caseProfile.value = (p && Object.keys(p).length) ? p : caseProfile.value
}
async function loadProfile(sid) {
  if (!sid) { caseProfile.value = null; return }
  try {
    const r = await fetch('/api/sessions/' + encodeURIComponent(sid) + '/profile', { headers: AUTH })
    const d = await r.json()
    caseProfile.value = (d.profile && Object.keys(d.profile).length) ? d.profile : null
  } catch (e) { caseProfile.value = null }
}

const examples = computed(() => {
  const r = userRole.value || 'resident'
  const map = {
    resident: [
      '楼上漏水导致我家天花板发霉怎么办',
      '邻居私装地锁占用公共车位，其他业主如何处理？',
      '家里反复发生肢体冲突，如何申请人身安全保护？',
    ],
    property: [
      '业主因为漏水投诉到物业，我们物业在调解里具体要承担什么职责？',
      '邻里噪音这种事，物业有没有强制执法权？还是只能协调？',
      '业主拒交物业费，我们物业该怎么处理？',
    ],
    mediator: [
      '调解邻里纠纷一般分哪几个步骤？首次上门要注意什么？',
      '调解达成协议后对方反悔不履行，有没有法律约束力？',
      '遇到当事人情绪激动甚至动手的情况，调解员怎么处置？',
    ],
    grid_worker: [
      '网格员走访发现独居老人多日未出门，该怎么处理？',
      '小区里发现疑似家暴情况，网格员第一时间该做什么？',
      '居民反映的垃圾分类问题一直没解决，怎么向上级反馈推动？',
    ],
  }
  return map[r] || map.resident
})

// 欢迎语与副标题：根据选中角色动态切换
const welcomeText = computed(() => {
  const r = userRole.value || 'resident'
  const map = {
    resident: { title: '你好，我是社区矛盾调解助理', sub: '描述你遇到的邻里 / 物业 / 家庭矛盾，我会结合知识库给出处置建议、相关法条与调解步骤，并标注依据来源。' },
    property: { title: '你好，我是社区矛盾调解助理', sub: '你是物业工作人员视角。描述遇到的业主投诉 / 邻里纠纷 / 设施争议，我会从物业职责角度给出处置建议与操作步骤。' },
    mediator: { title: '你好，我是社区矛盾调解助理', sub: '你是调解员 / 居委会视角。描述正在处理的矛盾纠纷，我会按调解流程给出处置建议、法律依据与话术指引。' },
    grid_worker: { title: '你好，我是社区矛盾调解助理', sub: '你是网格员 / 社工视角。描述走访中发现的问题或居民诉求，我会给出一线上门处置的行动指引与联动方案。' },
  }
  return map[r] || map.resident
})

onMounted(async () => {
  // 注册本会话提问下拉的全局监听（在 onMounted 里注册，避免 HMR / SSR 反复创建）
  document.addEventListener('click', onDocClickForHistory)
  document.addEventListener('keydown', onDocKeyForHistory)
  try {
    const r = await fetch('/api/models')
    const data = await r.json()
    models.value = data.models || []
    currentModel.value = data.default || 'deepseek'
  } catch (e) {
    console.warn('获取模型列表失败', e)
  }
  // 加载历史会话列表
  await refreshSessions()
})

// 点击消息锚点：滚动定位 + 更新 URL hash（WorkBuddy 式深链定位）
function locateMsg(i) {
  const el = document.getElementById('msg-' + i)
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  history.replaceState(null, '', '#msg-' + i)
  el.classList.remove('msg-flash')
  // 触发重排以重启动画
  void el.offsetWidth
  el.classList.add('msg-flash')
  setTimeout(() => el.classList.remove('msg-flash'), 1300)
}

// ---------- 本会话提问下拉（WorkBuddy 式：右上角按钮 → 列出用户提问 → 跳转） ----------
const showHistoryMenu = ref(false)
function toggleHistoryMenu() {
  showHistoryMenu.value = !showHistoryMenu.value
}
function closeHistoryMenu() { showHistoryMenu.value = false }
// 点击外部关闭：用 closest 判断是否点在 .history-menu 内（关键——之前 document 全局监听抢跑 toggle，导致点不开）
function onDocClickForHistory(e) {
  if (!showHistoryMenu.value) return
  const t = e && e.target
  if (t && t.closest && t.closest('.history-menu')) return
  showHistoryMenu.value = false
}
function onDocKeyForHistory(e) { if (e.key === 'Escape') closeHistoryMenu() }

const userQuestions = computed(() => {
  return messages
    .map((m, i) => ({ m, i }))
    .filter(x => x.m.role === 'user' && x.m.content)
    .map(x => {
      const t = (x.m.content || '').replace(/\s+/g, ' ').trim()
      return { i: x.i, text: t, preview: t.length > 14 ? t.slice(0, 14) + '…' : t }
    })
})
function jumpToUserQuestion(i) {
  closeHistoryMenu()
  locateMsg(i)
}

async function send(text) {
  const question = (text ?? input.value).trim()
  if (!question || loading.value) return
  if (!userRole.value) { alert('请先选择你的身份（居民 / 物业 / 调解员 / 网格员），再开始对话'); return }
  input.value = ''
  messages.push({ role: 'user', content: question })
  messages.push({ role: 'bot', content: '', loading: true })
  loading.value = true
  await nextTick()
  scrollToBottom()

  try {
    // 短期记忆：会话历史由后端按 session_id 拉取，前端不再拼接 history
    // stream=true 走 SSE：首字即可见，体感远快于等整段
    const sid = await ensureSession()
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 60000) // 60s 超时保护（含 LLM 耗时）
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-Id': userId },
      body: JSON.stringify({ question, provider: currentModel.value, session_id: sid, stream: true, user_role: userRole.value || undefined }),
      signal: controller.signal,
    })
    clearTimeout(timer)
    // 非 200 响应（护栏拦截/限流等）：直接读 JSON 错误体，不走 SSE
    if (!resp.ok) {
      let errBody = { error: `请求失败（HTTP ${resp.status}）` }
      try { errBody = await resp.json() } catch {}
      // 护栏拦截：给用户友好提示
      const msg = errBody.error || errBody.message || errBody
      const friendlyMsg = (resp.status === 400 && String(msg).includes('不安全'))
        ? '⚠️ 该请求被安全护栏拦截（可能包含注入指令），已拒绝处理。' : `⚠️ ${msg}`
      messages[messages.length - 1].content = friendlyMsg
      messages[messages.length - 1].loading = false
      loading.value = false
      return
    }
    const last = messages[messages.length - 1]
    last.loading = false
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    let acc = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') continue
        let ev
        try { ev = JSON.parse(payload) } catch { continue }
        if (ev.type === 'delta') {
          acc += ev.text
          last.content = acc
        } else if (ev.type === 'route') {
          last.route = ev.route
          last.traceId = ev.trace_id
          if (ev.session_id) sessionId.value = ev.session_id
        } else if (ev.type === 'done') {
          last.content = ev.answer
          last.route = ev.route
          last.sources = (ev.sources || []).filter(s => s.score >= SOURCE_MIN_SCORE)
          last.retries = ev.self_rag_retries
          last.model = ev.model
          last.latency = ev.latency_ms
          last.trace = ev.trace
          last.profile = ev.case_profile || null
          last.decomposition = ev.decomposition || null
          last.traceId = ev.trace_id
          if (ev.session_id) sessionId.value = ev.session_id
          updateProfile(ev.case_profile)
        } else if (ev.type === 'error') {
          last.content = '请求失败：' + (ev.error || '')
        }
        await nextTick()
        scrollToBottom()
      }
    }
    last.loading = false
    refreshSessions()
  } catch (e) {
    const last = messages[messages.length - 1]
    last.loading = false
    if (e.name === 'AbortError') {
      last.content = '⚠️ 请求超时（60秒无响应），请稍后重试。'
    } else {
      last.content = '请求失败：' + (e.message || e)
    }
  } finally {
    loading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  if (chatArea.value) chatArea.value.scrollTop = chatArea.value.scrollHeight
}

// 将回答中的 [N] 引用标记渲染为悬浮 tooltip（显示来源完整信息：标题/类别/相关度/摘要/法条）
function renderAnswer(text, sources) {
  if (!text) return ''
  // 安全转义 HTML 特殊字符
  let safe = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  // 将 [N] 替换为带 data 属性的 span，CSS :hover 显示 tooltip
  if (sources && sources.length) {
    safe = safe.replace(/\[(\d+)\]/g, (match, num) => {
      const idx = parseInt(num) - 1
      const s = sources[idx]
      if (!s) return match
      const lines = [
        `📄 ${s.title}`,
        `━━━━━━━━━━━━━━━`,
        `类别：${s.category || '未知'}  |  相关度：${s.score || '-'}`,
      ]
      if (s.content) {
        let summary = s.content.slice(0, 300)
        if (s.content.length > 300) {
          const lastPeriod = Math.max(summary.lastIndexOf('。'), summary.lastIndexOf('；'), summary.lastIndexOf('.'))
          if (lastPeriod > 100) summary = summary.slice(0, lastPeriod + 1)
          else summary += '…'
        }
        lines.push(``, `📝 摘要：`, summary)
      }
      if (s.legal_basis) lines.push(``, `⚖️ 法条：${s.legal_basis}`)
      const tip = lines.join('\n')
      return `<span class="ref-tag" data-tip="${tip.replace(/"/g, '&quot;')}">[${num}]</span>`
    })
  }
  return safe
}

// ---------- 会话管理（短期记忆） ----------
function fmtTime(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

async function refreshSessions() {
  try {
    const r = await fetch('/api/sessions', { headers: AUTH })
    const d = await r.json()
    sessions.value = d.sessions || []
  } catch (e) { console.warn('获取会话列表失败', e) }
}

async function ensureSession() {
  if (sessionId.value) return sessionId.value
  // 首次发消息自动建会话
  const r = await fetch('/api/sessions', { method: 'POST', headers: AUTH })
  const d = await r.json()
  sessionId.value = d.session_id
  await refreshSessions()
  return sessionId.value
}

// 新建会话：清屏并新建一段会话上下文（后端持久化）
async function newChat() {
  if (loading.value) return
  messages.splice(0, messages.length)
  input.value = ''
  caseProfile.value = null
  userRole.value = ''  // 重置身份选择，让用户重新选
  try {
    const r = await fetch('/api/sessions', { method: 'POST', headers: AUTH })
    const d = await r.json()
    sessionId.value = d.session_id
    await refreshSessions()
  } catch (e) { console.warn('新建会话失败', e); sessionId.value = '' }
}

// 打开历史会话：加载完整消息到主区域
async function loadSession(id) {
  if (loading.value) return
  try {
    const r = await fetch('/api/sessions/' + encodeURIComponent(id), { headers: AUTH })
    const d = await r.json()
    sessionId.value = id
    messages.splice(0, messages.length)
    for (const m of (d.messages || [])) {
      messages.push({
        role: m.role === 'assistant' ? 'bot' : 'user',
        content: m.content,
      })
    }
    await loadProfile(id)
  } catch (e) { console.warn('加载会话失败', e) }
}

// 删除会话
async function deleteSession(id) {
  if (!confirm('确定删除该会话？此操作不可恢复。')) return
  try {
    await fetch('/api/sessions/' + encodeURIComponent(id), { method: 'DELETE', headers: AUTH })
    if (sessionId.value === id) {
      sessionId.value = ''
      messages.splice(0, messages.length)
      caseProfile.value = null
    }
    await refreshSessions()
  } catch (e) { console.warn('删除会话失败', e) }
}

const routeLabel = { retrieve: '检索回答', direct: '直接回答', clarify: '需澄清', out_of_domain: '超出范围' }
const roleLabel2 = { resident: '居民/业主', mediator: '调解员/居委会', property: '物业人员', grid_worker: '网格员/社工' }

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
    <!-- 左侧会话栏：历史会话列表（短期记忆） -->
    <aside class="sidebar" v-if="tab === 'chat'">
      <div class="sidebar-head">
        <button class="new-chat full" :disabled="loading" @click="newChat">＋ 新建会话</button>
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          :class="['session-item', { active: s.id === sessionId }]"
          @click="loadSession(s.id)"
        >
          <div class="session-main">
            <div class="session-title">{{ s.title || '新会话' }}</div>
            <div class="session-meta">{{ fmtTime(s.updated_at) }}</div>
          </div>
          <button class="session-del" @click.stop="deleteSession(s.id)" title="删除会话">✕</button>
        </div>
        <div v-if="!sessions.length" class="session-empty">暂无历史会话，点上方新建</div>
      </div>
    </aside>

    <div class="main">
      <header class="app-header">
        <h1>🤝 社区矛盾调解 RAG 助手 <span class="badge">Agentic RAG</span></h1>
        <div class="header-bar">
          <nav class="tabs">
            <button :class="['tab', { active: tab === 'chat' }]" @click="switchTab('chat')">对话</button>
            <button :class="['tab', { active: tab === 'kb' }]" @click="switchTab('kb')">知识库</button>
          </nav>
          <div class="header-controls" v-if="tab === 'chat'">
            <div class="history-menu">
              <button class="history-btn" :class="{ active: showHistoryMenu }"
                      @click="toggleHistoryMenu" :title="`本会话提问（${userQuestions.length}）`">
                📋 本会话提问 <span class="badge-count" v-if="userQuestions.length">{{ userQuestions.length }}</span>
                <span class="caret" :class="{ open: showHistoryMenu }">▾</span>
              </button>
              <div class="history-dropdown" v-if="showHistoryMenu" @click.stop>
                <div class="history-dropdown-head">
                  <span>本会话用户提问（{{ userQuestions.length }}）</span>
                  <button class="history-close" @click="showHistoryMenu = false" title="关闭">×</button>
                </div>
                <div class="history-list" v-if="userQuestions.length">
                  <button v-for="q in userQuestions" :key="q.i"
                          class="history-item" @click="jumpToUserQuestion(q.i)">
                    <span class="history-idx">#{{ q.i + 1 }}</span>
                    <span class="history-text" :title="q.text">{{ q.preview }}</span>
                  </button>
                </div>
                <div class="history-empty" v-else>本会话还没有提问</div>
              </div>
            </div>
            <select class="model-select" v-model="currentModel">
              <option v-for="m in models" :key="m.provider" :value="m.provider">
                {{ m.label }}（{{ m.model }}）{{ m.available ? '' : '· 未配置key' }}
              </option>
            </select>
          </div>
        </div>
      </header>

      <!-- 长期记忆可视化：本次会话沉淀的结构化案件档案 -->
      <div v-if="tab === 'chat' && caseProfile" class="profile-bar">
        <button class="profile-toggle" @click="profileOpen = !profileOpen">
          📋 案件档案 <span class="profile-caret">{{ profileOpen ? '▾' : '▸' }}</span>
        </button>
        <div class="profile-chips" v-if="!profileOpen">
          <span class="pc" v-if="caseProfile.case_type">{{ caseProfile.case_type }}</span>
          <span class="pc" v-if="caseProfile.opponent">{{ caseProfile.opponent }}</span>
          <span class="pc" v-if="caseProfile.stage">{{ caseProfile.stage }}</span>
        </div>
        <div class="profile-detail" v-if="profileOpen">
          <div class="pf-row" v-if="caseProfile.case_type"><b>案件类型</b><span>{{ caseProfile.case_type }}</span></div>
          <div class="pf-row" v-if="caseProfile.opponent"><b>对方当事人</b><span>{{ caseProfile.opponent }}</span></div>
          <div class="pf-row" v-if="caseProfile.identity"><b>用户身份</b><span>{{ roleLabel2[caseProfile.identity] || caseProfile.identity }}</span></div>
          <div class="pf-row" v-if="caseProfile.stage"><b>当前阶段</b><span>{{ caseProfile.stage }}</span></div>
          <div class="pf-row" v-if="caseProfile.evidence_status"><b>证据情况</b><span>{{ caseProfile.evidence_status }}</span></div>
          <div class="pf-row" v-if="caseProfile.key_facts && Object.keys(caseProfile.key_facts).length">
            <b>已知事实</b>
            <span class="kf-list">
              <span class="kf" v-for="(v, k) in caseProfile.key_facts" :key="k">{{ k }}：{{ v }}</span>
            </span>
          </div>
        </div>
      </div>

    <div v-if="tab === 'chat'" class="chat-area" ref="chatArea">
      <div v-if="messages.length === 0" class="empty">
        <!-- 未选角色时：全屏角色选择器，遮挡其他内容 -->
        <div v-if="showRolePicker" class="role-overlay">
          <h2>👋 欢迎使用社区矛盾调解助手</h2>
          <p class="role-overlay-sub">请先选择你的身份，我会据此调整回答视角和推荐问题</p>
          <div class="role-options">
            <button
              v-for="r in ROLE_OPTIONS" :key="r.value"
              :class="['role-opt', { active: userRole === r.value }]"
              @click="userRole = r.value"
            >
              <span class="role-opt-label">{{ r.label }}</span>
              <span class="role-opt-desc">{{ r.desc }}</span>
            </button>
          </div>
        </div>

        <!-- 已选角色后：显示欢迎语 + 角色化推荐问题 -->
        <template v-else>
          <h2>{{ welcomeText.title }}</h2>
          <p>{{ welcomeText.sub }}</p>
          <div class="chips">
            <span class="chip" v-for="ex in examples" :key="ex" @click="send(ex)">{{ ex }}</span>
          </div>
        </template>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role" :id="'msg-' + i">
        <div class="avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
        <div class="bubble">
          <button class="msg-anchor" :title="'点击定位此条 #' + (i + 1)" @click="locateMsg(i)">#{{ i + 1 }}</button>
          <div v-if="m.role === 'bot' && m.route" class="route-tag">
            {{ routeLabel[m.route] || m.route }} · 模型 {{ m.model }}
            <span v-if="m.retries"> · 自纠错重试 {{ m.retries }} 次</span>
          </div>
          <div v-if="m.decomposition && m.decomposition.enabled" class="decomp-box">
            <span class="decomp-title">🧩 查询已自动拆分（{{ m.decomposition.sub_queries.length }} 个子问题分别检索后合并）</span>
            <ol class="decomp-list">
              <li v-for="(sq, di) in m.decomposition.sub_queries" :key="di">{{ sq }}</li>
            </ol>
          </div>
          <div v-if="m.loading" class="typing">正在检索知识库并生成…</div>
          <div v-else class="answer-body" v-html="renderAnswer(m.content, m.sources)"></div>

          <!-- 来源：可展开卡片（点击展开完整内容） -->
          <div v-if="m.sources && m.sources.length" class="sources-compact">
            <span class="sources-label">📎 相关资料（{{ m.sources.length }} 篇）</span>
            <template v-for="s in m.sources" :key="s.id">
              <details class="source-detail">
                <summary>{{ s.title }}</summary>
                <div class="source-body">
                  <div class="source-meta">{{ s.category }} · 相关度 {{ s.score }}</div>
                  <div class="source-content" v-if="s.content">{{ s.content }}</div>
                  <div class="source-law" v-if="s.legal_basis">⚖️ {{ s.legal_basis }}</div>
                </div>
              </details>
            </template>
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
  </div>
</template>

<style scoped>
.msg {
  position: relative;
}
.msg-anchor {
  position: absolute;
  top: 8px;
  right: 10px;
  z-index: 2;
  min-width: 26px;
  height: 22px;
  padding: 0 6px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #fff;
  color: #94a3b8;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height: 20px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.msg:hover .msg-anchor {
  opacity: 1;
}
.msg-anchor:hover {
  color: #2563eb;
  border-color: #2563eb;
}
.msg-flash {
  animation: msgFlash 1.3s ease;
}
@keyframes msgFlash {
  0%   { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
  20%  { box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.35); }
  100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

/* ---------- 头部：本会话提问下拉 ---------- */
.history-menu { position: relative; }
.history-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 12px; border: 1px solid #d6dde6; border-radius: 8px;
  background: #fff; color: #2d3748; font-size: 13px; cursor: pointer;
  transition: all 0.15s;
}
.history-btn:hover { border-color: #5a7ad8; color: #2c5282; }
.history-btn.active { border-color: #5a7ad8; background: #eef3ff; color: #2c5282; }
.badge-count {
  display: inline-block; min-width: 18px; padding: 0 5px; line-height: 16px;
  background: #5a7ad8; color: #fff; border-radius: 9px; font-size: 11px; text-align: center;
}
.caret { font-size: 10px; color: #718096; transition: transform 0.15s; }
.caret.open { transform: rotate(180deg); }

.history-dropdown {
  position: absolute; top: calc(100% + 6px); right: 0; width: 320px;
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  box-shadow: 0 8px 28px rgba(0,0,0,0.12); z-index: 1000;
  max-height: 70vh; display: flex; flex-direction: column; overflow: hidden;
}
.history-dropdown-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; border-bottom: 1px solid #edf2f7;
  font-size: 12px; color: #4a5568; font-weight: 600;
}
.history-close {
  background: transparent; border: none; cursor: pointer;
  font-size: 18px; color: #a0aec0; line-height: 1; padding: 0 4px;
}
.history-close:hover { color: #2d3748; }
.history-list { overflow-y: auto; padding: 4px 0; }
.history-item {
  display: flex; align-items: center; gap: 8px; width: 100%;
  padding: 8px 14px; background: transparent; border: none; cursor: pointer;
  text-align: left; font-size: 13px; color: #2d3748;
  transition: background 0.12s;
}
.history-item:hover { background: #f7fafc; }
.history-idx {
  flex-shrink: 0; min-width: 32px; font-size: 11px; color: #718096;
  font-family: ui-monospace, SFMono-Regular, monospace;
}
.history-text {
  flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.history-empty {
  padding: 24px 14px; text-align: center; color: #a0aec0; font-size: 13px;
}

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
.decomp-box {
  margin: 8px 0 4px;
  padding: 8px 10px;
  background: #eef5ff;
  border: 1px solid #cfe0ff;
  border-radius: 8px;
}
.decomp-title {
  display: block;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
}
.decomp-list {
  margin: 0;
  padding-left: 18px;
}
.decomp-list li {
  color: #334155;
  font-size: 12.5px;
  line-height: 1.6;
}
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
