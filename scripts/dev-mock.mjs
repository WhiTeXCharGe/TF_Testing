// Runs the local "mock Azure" stack: ACA1 (session API, :4000) + ACA2 (live
// relay, :4010) + the Vite web client (:5173), all sharing ./server/mock-blob
// as stand-in Blob storage. Zero extra dependencies — plain child_process.
//
//   npm run dev:mock
//
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// The web client learns where ACA1 is from config.txt (served by the local
// server on :3010 via the Vite /api proxy) — point it at the mock ACA1.
const MOCK_CONFIG = join(tmpdir(), 'gantt-dev-mock-config.txt');
writeFileSync(MOCK_CONFIG, 'mode=online\nazure_url=http://localhost:4000\n', 'utf-8');

const SHARED = {
  STORAGE: 'fs',
  MOCK_BLOB_DIR: './mock-blob', // relative to server/ (npm --prefix server)
  INTERNAL_KEY: 'mock-internal-key',
  WEB_ORIGIN: 'http://localhost:5173',
};

const targets = [
  ['aca1', 'npm', ['--prefix', 'server', 'run', 'dev'], { ...SHARED, ROLE: 'aca1', PORT: '4000', ACA2_URL: 'http://localhost:4010' }],
  ['aca2', 'npm', ['--prefix', 'server', 'run', 'dev'], { ...SHARED, ROLE: 'aca2', PORT: '4010', PUBLIC_RELAY_URL: 'http://localhost:4010' }],
  // Local server on :3010 (what Vite proxies /api to): serves /api/app-config
  // from the file above, so the dev client talks to the mock ACA1 on :4000.
  ['local', 'npm', ['--prefix', 'server', 'run', 'dev'], { ROLE: 'local', PORT: '3010', APP_CONFIG_PATH: MOCK_CONFIG }],
  ['web', 'npm', ['run', 'dev'], {}],
];

let shuttingDown = false;
const children = [];

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const c of children) {
    try { c.kill(); } catch { /* already gone */ }
  }
  process.exit(code);
}

for (const [name, cmd, args, env] of targets) {
  const child = spawn(cmd, args, {
    env: { ...process.env, ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32',
  });
  const log = (buf) => {
    for (const line of buf.toString().split('\n')) {
      if (line.trim()) console.log(`[${name}] ${line}`);
    }
  };
  child.stdout.on('data', log);
  child.stderr.on('data', log);
  child.on('exit', (c) => {
    console.log(`[${name}] exited with ${c}`);
    shutdown(c ?? 0);
  });
  children.push(child);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
console.log('[dev:mock] ACA1 http://localhost:4000  ·  ACA2 http://localhost:4010  ·  web http://localhost:5173');
