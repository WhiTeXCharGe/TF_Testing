// Runs the REAL sigmai-setup/bridge.sh against the fake Sigma-I server (no Docker/Azure).
// Needs bash + curl + python (Git Bash on Windows: run `npm test` from Git Bash so `bash` is Git's).
// Cancel (SIGTERM) can't be sent on Windows from node — covered by documents/SigmaI/SigmaI-04-Local-Testing.md.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const posix = (p) => p.split(path.sep).join('/');
const BRIDGE = posix(path.resolve(HERE, '../sigmai-setup/bridge.sh'));
const BASH = process.env.BASH_BIN || 'bash';

async function startFake(mode, polls = 2) {
  const { createFakeSigmai } = await import('./fake-sigmai-server.mjs');
  const app = createFakeSigmai({ mode, polls });
  const server = http.createServer(app);
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return { app, server, url: `http://127.0.0.1:${server.address().port}` };
}

function makeWork(runId, options) {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'bridge-'));
  const input = path.join(work, 'input', 'input', runId);
  fs.mkdirSync(input, { recursive: true });
  fs.writeFileSync(path.join(input, 'EnvConfig.yaml'), 'environment:\n  worker_list: []\n');
  fs.writeFileSync(path.join(input, 'Schedule.yaml'), 'schedule:\n  plan_range: {start_date: "2026-01-01", end_date: "2026-01-31"}\n');
  fs.writeFileSync(path.join(input, 'sigmai_options.json'), JSON.stringify({
    engine: 'sigmai', stage1Solver: 'scip', stage2Solver: 'scip', stage1TimeoutSec: 30, stage2TimeoutSec: 30,
    stage1NumSolutionsLimit: null, stage2NumSolutionsLimit: null, weightPreset: 'standard',
    disabledConstraints: ['stage2.stay_limits'], scenarioName: '@weird;label', ...options,
  }));
  return work;
}

function runBridge({ runId, work, url, extraEnv = {} }) {
  return new Promise((resolve) => {
    const p = spawn(BASH, [BRIDGE], {
      env: {
        ...process.env, RUN_ID: runId, BRIDGE_WORK_DIR: posix(work),
        SIGMAI_API_URL: url, SIGMAI_API_KEY: 'test-key', SIGMAI_HOME: posix(path.join(work, 'home')),
        POLL_INTERVAL_SEC: '1', ...extraEnv,
      },
    });
    let out = '';
    p.stdout.on('data', (d) => (out += d)); p.stderr.on('data', (d) => (out += d));
    p.on('close', (code) => resolve({ code, out }));
  });
}
const readStatus = (work, id) => JSON.parse(fs.readFileSync(path.join(work, 'status', `${id}.json`), 'utf8'));

test('bridge: successful run (option ③)', async () => {
  const { server, url, app } = await startFake('ok');
  const work = makeWork('run_ok');
  const { code, out } = await runBridge({ runId: 'run_ok', work, url });
  server.close();
  assert.equal(code, 0, out);
  const st = readStatus(work, 'run_ok');
  assert.equal(st.status, 'Completed');
  assert.equal(st.engine, 'sigmai');
  assert.equal(st.option, 'sigmai-scip-scip');
  assert.equal(st.progress, 100);
  assert.equal(st.sigmai.summary.validationPassed, true);
  assert.match(fs.readFileSync(path.join(work, 'output', 'result_Schedule.yaml'), 'utf8'), /# result for run_ok/);
  assert.ok(fs.existsSync(path.join(work, 'output', 'sigmai', 'result.json')));
  assert.ok(fs.existsSync(path.join(work, 'output', 'sigmai', 'bridge.log')));
  // form fields reached Sigma-I with the right names, and "@weird" was NOT treated as a file
  const f = app.jobs.get('run_ok').fields;
  assert.equal(f.stage1_solver, 'scip');
  assert.equal(f.stage1_timeout, '30');
  assert.equal(f.scenario_name, '@weird;label');
  assert.equal(f.disabled_constraints, '["stage2.stay_limits"]');
  assert.equal(f.weight_preset, 'standard');
});

test('bridge: Sigma-I failure → Failed with error type, no result file', async () => {
  const { server, url } = await startFake('fail');
  const work = makeWork('run_fail');
  const { code } = await runBridge({ runId: 'run_fail', work, url });
  server.close();
  assert.equal(code, 1);
  const st = readStatus(work, 'run_fail');
  assert.equal(st.status, 'Failed');
  assert.equal(st.error.type, 'OPTIMIZATION_FAILED');
  assert.equal(fs.existsSync(path.join(work, 'output', 'result_Schedule.yaml')), false);
});

test('bridge: wrong API key → InvalidInput (4xx is not retried)', async () => {
  const { server, url } = await startFake('ok');
  const work = makeWork('run_auth');
  const { code } = await runBridge({ runId: 'run_auth', work, url, extraEnv: { SIGMAI_API_KEY: 'nope' } });
  server.close();
  assert.equal(code, 1);
  assert.equal(readStatus(work, 'run_auth').error.type, 'InvalidInput');
});

test('bridge: Sigma-I unreachable → SigmaIUnavailable', async () => {
  const work = makeWork('run_down');
  const { code } = await runBridge({ runId: 'run_down', work, url: 'http://127.0.0.1:9' });
  assert.equal(code, 1);
  assert.equal(readStatus(work, 'run_down').error.type, 'SigmaIUnavailable');
});

test('bridge: missing input → InvalidInput', async () => {
  const work = makeWork('run_noin');
  fs.rmSync(path.join(work, 'input', 'input', 'run_noin', 'Schedule.yaml'));
  const { code } = await runBridge({ runId: 'run_noin', work, url: 'http://127.0.0.1:9' });
  assert.equal(code, 1);
  assert.equal(readStatus(work, 'run_noin').error.type, 'InvalidInput');
});

test('bridge: deadline exceeded → TIMEOUT and job deleted', async () => {
  const { server, url, app } = await startFake('slow');
  const work = makeWork('run_slow', { stage1TimeoutSec: 1, stage2TimeoutSec: 1 });
  const { code } = await runBridge({
    runId: 'run_slow', work, url, extraEnv: { DEADLINE_MARGIN_SEC: '2', POLL_INTERVAL_SEC: '1' },
  });
  server.close();
  assert.equal(code, 1);
  assert.equal(readStatus(work, 'run_slow').error.type, 'TIMEOUT');
  assert.equal(app.jobs.get('run_slow').canceled, true);
});

test('bridge: job already exists (task retried) → resumes polling', async () => {
  const { server, url, app } = await startFake('ok');
  app.jobs.set('run_dup', { fields: {}, schedule: 'x: 1\n', polls: 5, canceled: false });
  const work = makeWork('run_dup');
  const { code, out } = await runBridge({ runId: 'run_dup', work, url });
  server.close();
  assert.equal(code, 0, out);
  assert.equal(readStatus(work, 'run_dup').status, 'Completed');
});
