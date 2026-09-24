import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Function form (not a plain object) so we can call loadEnv() — plain
// `process.env.VITE_ACA1_URL` only sees a *real* shell env var (e.g. an
// `export` before the build), never a `.env*` file, because Vite's automatic
// .env loading only populates `import.meta.env` for the client bundle, not
// this Node-side config file. loadEnv() reads the same .env*/.env.production
// files explicitly so `.env.production` (checked in, see that file's own
// comment) bakes in the deployed Azure URL for `npm run electron:build`
// without needing anyone to `export` it by hand first.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    // Baked in at build time; empty in dev/LAN so collabService falls back to
    // the current host on the ACA1 port. The packaged desktop app still
    // probes this URL at runtime (collabService.probeAzureReachability) and
    // falls back to its own local server if Azure can't be reached.
    define: {
      __ACA1_URL__: JSON.stringify(env.VITE_ACA1_URL ?? ''),
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
  };
});