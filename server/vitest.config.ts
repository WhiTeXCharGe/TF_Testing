import { defineConfig } from 'vitest/config';

// Without a config file here, Vitest walks up and inherits the CLIENT's
// vite.config.ts at the repo root, which has no idea about this package —
// notably no exclusion for server/dist, so once `npm run build` has ever run,
// the stale compiled copies under dist/collab/*.test.js were collected
// alongside the real src/collab/*.test.ts and every test ran twice (a false
// green waiting to happen, since the dist copies are whatever the last build
// produced). Scope discovery to the TypeScript sources only.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['dist/**', 'node_modules/**'],
    environment: 'node',
  },
});
