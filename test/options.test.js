import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseRunOptions, buildSigmaiOptionsFile, wallClockDuration, queuePositionFor,
  buildManifest, finalizeManifest, bridgeExitedWithoutStatus, isValidRunId,
} from '../src/options.js';

const ON  = { sigmaiEnabled: true,  dwaveEnabled: true,  maxStageTimeoutSec: 7200 };
const OFF = { sigmaiEnabled: false, dwaveEnabled: false, maxStageTimeoutSec: 7200 };

test('no engine field → timefold (v1 compatible)', () => {
  const r = parseRunOptions({ runId: 'x' }, OFF);
  assert.equal(r.ok, true);
  assert.equal(r.value.engine, 'timefold');
  assert.equal(r.value.option, 'timefold');
  assert.equal(r.value.solver, null);
});

test('engine=timefold rejects Sigma-I fields', () => {
  const r = parseRunOptions({ engine: 'timefold', stage1Solver: 'scip' }, ON);
  assert.equal(r.ok, false);
  assert.match(r.error, /stage1Solver/);
});

test('unknown engine rejected', () => {
  assert.equal(parseRunOptions({ engine: 'cplex' }, ON).ok, false);
});

test('sigmai refused when not enabled', () => {
  const r = parseRunOptions({ engine: 'sigmai' }, OFF);
  assert.equal(r.ok, false);
  assert.match(r.error, /not enabled/);
});

test('option ② dwave+scip with UI fields', () => {
  const r = parseRunOptions({
    engine: 'sigmai', stage1Solver: 'dwave', stage2Solver: 'scip',
    stage1TimeoutSec: '600', stage2TimeoutSec: '1800', label: 'Plan\nSep',
  }, ON);
  assert.equal(r.ok, true);
  assert.equal(r.value.option, 'sigmai-dwave-scip');
  assert.equal(r.value.label, 'Plan Sep');
  assert.deepEqual(
    [r.value.solver.stage1TimeoutSec, r.value.solver.stage2TimeoutSec], [600, 1800]);
});

test('option ③ scip+scip, defaults when timeouts omitted', () => {
  const r = parseRunOptions({ engine: 'sigmai', stage1Solver: 'scip', stage2Solver: 'scip' }, ON);
  assert.equal(r.ok, true);
  assert.equal(r.value.option, 'sigmai-scip-scip');
  assert.equal(r.value.solver.stage1TimeoutSec, 1800);
});

test('dwave refused when D-Wave disabled', () => {
  const r = parseRunOptions({ engine: 'sigmai', stage1Solver: 'dwave' }, { ...ON, dwaveEnabled: false });
  assert.equal(r.ok, false);
  assert.match(r.error, /D-Wave/);
});

test('stage2 must be scip', () => {
  assert.equal(parseRunOptions({ engine: 'sigmai', stage2Solver: 'dwave' }, ON).ok, false);
});

test('timeout bounds', () => {
  for (const bad of ['0', '-5', '1.5', 'abc', '7201']) {
    assert.equal(parseRunOptions({ engine: 'sigmai', stage1TimeoutSec: bad }, ON).ok, false, bad);
  }
  assert.equal(parseRunOptions({ engine: 'sigmai', stage1TimeoutSec: '7200' }, ON).ok, true);
});

test('weightPreset and disabledConstraints validation', () => {
  assert.equal(parseRunOptions({ engine: 'sigmai', weightPreset: 'nope' }, ON).ok, false);
  assert.equal(parseRunOptions({ engine: 'sigmai', disabledConstraints: 'not json' }, ON).ok, false);
  assert.equal(parseRunOptions({ engine: 'sigmai', disabledConstraints: '["x"]' }, ON).ok, false);
  assert.equal(parseRunOptions({ engine: 'sigmai', disabledConstraints: '["stage2.stay_limits","stage2.stay_limits"]' }, ON).ok, false);
  const ok = parseRunOptions({
    engine: 'sigmai', weightPreset: 'daily_balance', disabledConstraints: '["stage2.stay_limits"]',
  }, ON);
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.value.solver.disabledConstraints, ['stage2.stay_limits']);
});

test('label > 200 chars rejected', () => {
  assert.equal(parseRunOptions({ label: 'a'.repeat(201) }, ON).ok, false);
});

test('sigmai_options.json shape', () => {
  const r = parseRunOptions({ engine: 'sigmai', stage1Solver: 'scip', label: 'L' }, ON);
  const f = buildSigmaiOptionsFile(r.value);
  assert.equal(f.engine, 'sigmai');
  assert.equal(f.stage1Solver, 'scip');
  assert.equal(f.scenarioName, 'L');
  assert.deepEqual(f.disabledConstraints, []);
});

test('wallClockDuration = stages + margin', () => {
  assert.equal(wallClockDuration({ stage1TimeoutSec: 600, stage2TimeoutSec: 1800 }, 900), 'PT0H55M0S');
  assert.equal(wallClockDuration({ stage1TimeoutSec: 1, stage2TimeoutSec: 1 }, 0), 'PT0H0M2S');
});

test('queuePositionFor counts older active tasks', () => {
  const tasks = [
    { id: 'a', creationTime: '2026-10-20T01:00:00Z' },
    { id: 'b', creationTime: '2026-10-20T01:01:00Z' },
    { id: 'c', creationTime: '2026-10-20T01:02:00Z' },
  ];
  assert.equal(queuePositionFor('a', tasks[0].creationTime, tasks), 1);
  assert.equal(queuePositionFor('c', tasks[2].creationTime, tasks), 3);
});

test('runId validation', () => {
  assert.equal(isValidRunId('20261020_101010123'), true);
  assert.equal(isValidRunId('../x'), false);
  assert.equal(isValidRunId(''), false);
});

test('finalizeManifest is idempotent and fills execution/result', () => {
  const parsed = parseRunOptions({ engine: 'sigmai', stage1Solver: 'scip' }, ON).value;
  const m = buildManifest({
    runId: 'r1', parsed, nowIso: '2026-10-20T00:00:00Z', batchJobId: 'job-sigmai',
    inputInfo: {}, versions: {},
  });
  const task = {
    creationTime: '2026-10-20T00:00:00Z',
    executionInfo: { startTime: '2026-10-20T00:01:35Z', endTime: '2026-10-20T00:22:35Z', retryCount: 0 },
    nodeInfo: { poolId: 'pool-sigmai', nodeId: 'tvm-1' },
  };
  const f = finalizeManifest(m, { status: 'Completed', output: 'output/r1/result_Schedule.yaml' }, task, {
    summary: { validation_passed: true, final_score: 12.5 },
  });
  assert.equal(f.execution.queuedSec, 95);
  assert.equal(f.execution.runSec, 1260);
  assert.equal(f.execution.nodeId, 'tvm-1');
  assert.equal(f.result.sigmai.validationPassed, true);
  assert.equal(finalizeManifest(f, { status: 'Failed' }, null, null), f);
});

test('bridgeExitedWithoutStatus waits for the grace period', () => {
  const task = { executionInfo: { endTime: '2026-10-20T00:00:00Z', exitCode: 1 } };
  const base = { runId: 'r', status: 'Running' };
  const end = new Date('2026-10-20T00:00:00Z').getTime();
  assert.equal(bridgeExitedWithoutStatus(base, task, 120, end + 60_000), null);
  const f = bridgeExitedWithoutStatus(base, task, 120, end + 121_000);
  assert.equal(f.status, 'Failed');
  assert.equal(f.error.type, 'BridgeExitedWithoutStatus');
});
