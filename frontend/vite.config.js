import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时把 /api 代理到后端（默认 http://localhost:8000）
export default defineConfig({
  plugins: [vue()],
  server: {
    host: true, // 监听 0.0.0.0 + ::，使 127.0.0.1 与局域网/手机也能访问
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
