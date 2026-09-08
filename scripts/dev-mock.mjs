// Runs the local "mock Azure" stack: ACA1 (session API, :4000) + ACA2 (live
// relay, :4010) + the Vite web client (:5173), all sharing ./server/mock-blob
// as stand-in Blob storage. Zero extra dependencies — plain child_process.
//
//   npm run dev:mock
//
import { spawn } from 'node:child_process';

const SHARED = {
  STORAGE: 'fs',
  MOCK_BLOB_DIR: './mock-blob', // relative to server/ (npm --prefix server)
  INTERNAL_KEY: 'mock-internal-key',
  WEB_ORIGIN: 'http://localhost:5173',
};

const targets = [
  ['aca1', 'npm', ['--prefix', 'server', 'run', 'dev'], { ...SHARED, ROLE: 'aca1', PORT: '4000', ACA2_URL: 'http://localhost:4010' }],
  ['aca2', 'npm', ['--prefix', 'server', 'run', 'dev'], { ...SHARED, ROLE: 'aca2', PORT: '4010', PUBLIC_RELAY_URL: 'http://localhost:4010' }],
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
