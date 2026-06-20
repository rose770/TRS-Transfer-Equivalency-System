import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/equivalency': 'http://localhost:8000',
      '/courses': 'http://localhost:8000',
    }
  }
})