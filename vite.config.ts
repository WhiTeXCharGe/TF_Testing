import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    // Pre-bundle exceljs so Vite applies CJS→ESM interop (its dist is a UMD
    // bundle with no real ESM default export). Excluding it breaks the import.
    include: ['exceljs'],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
});
