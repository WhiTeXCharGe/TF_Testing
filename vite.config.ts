import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Which server the app talks to (local vs the Azure session server, and which
// Azure URL) is NOT decided here — it is read at runtime from config.txt next
// to the installed exe (served by the local server at /api/app-config), so one
// build serves both customers and developers.
export default defineConfig({
  plugins: [react()],
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
