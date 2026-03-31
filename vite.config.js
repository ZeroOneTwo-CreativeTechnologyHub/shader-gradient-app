import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Change 'shader-gradient-app' to your actual GitHub repo name
const repoName = 'shader-gradient-app'

export default defineConfig({
  plugins: [react()],
  // For GitHub Pages: /<repo-name>/ 
  // For custom domain or local dev, use '/'
  base: process.env.NODE_ENV === 'production' ? `/${repoName}/` : '/',
  server: {
    port: 5173,
    host: true,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
})