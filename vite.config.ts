import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Baked in at build time; empty in dev/LAN so collabService falls back to
  // the current host on the ACA1 port. CI sets VITE_ACA1_URL for the Azure build.
  define: {
    __ACA1_URL__: JSON.stringify(process.env.VITE_ACA1_URL ?? ''),
  },
  server: {
    host: true, // bind 0.0.0.0 so other PCs on the LAN can open a shared Live View link
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:3010',
        changeOrigin: true,
      },
    },
  },
});