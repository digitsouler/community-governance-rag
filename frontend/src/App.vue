<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'

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
  messages.push({ role: 'user', content: question })
  messages.push({ role: 'bot', content: '', loading: true })
  loading.value = true
  await nextTick()
  scrollToBottom()

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, provider: currentModel.value })
    })
    const data = await resp.json()
    const last = messages[messages.length - 1]
    last.loading = false
    last.content = data.answer || ''
    last.route = data.route
    last.sources = data.sources || []
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

const routeLabel = { retrieve: '检索回答', direct: '直接回答', clarify: '需澄清' }
</script>

<template>
  <div class="app">
    <header class="app-header">
      <h1>🤝 社区矛盾调解 RAG 助手 <span class="badge">Agentic RAG</span></h1>
      <select class="model-select" v-model="currentModel">
        <option v-for="m in models" :key="m.provider" :value="m.provider">
          {{ m.label }}（{{ m.model }}）{{ m.available ? '' : '· 未配置key' }}
        </option>
      </select>
    </header>

    <div class="chat-area" ref="chatArea">
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

    <div class="input-bar">
      <textarea
        v-model="input"
        rows="1"
        placeholder="描述矛盾事实，例如：一楼私装地锁占用公共车位…"
        @keydown.enter.exact.prevent="send()"
      ></textarea>
      <button class="send-btn" @click="send()" :disabled="loading">发送</button>
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
</style>
