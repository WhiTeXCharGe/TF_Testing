/**
 * Pure helpers (no Azure, no Express) — validation of /runSolver options and the
 * status/manifest bookkeeping. Kept separate from server.js so they can be unit tested
 * with `npm test` without any Azure resources.
 *
 * Contract: documents/SigmaI/SigmaI_Azure_DetailedDesign20261013.md §5.1, §6.4, §9.
 */

export const RUN_ID_RE = /^[A-Za-z0-9_-]+$/;
export const WEIGHT_PRESETS = ['standard', 'daily_balance', 'worker_balance'];
export const DISABLED_CONSTRAINT_IDS = [
  'stage2.overtime_limits',
  'stage2.stay_limits',
  'stage2.manager_assignment',
];
export const TERMINAL = ['Completed', 'Failed', 'Cancelled'];

const SIGMAI_FIELDS = [
  'stage1Solver', 'stage2Solver', 'stage1TimeoutSec', 'stage2TimeoutSec',
  'stage1NumSolutionsLimit', 'stage2NumSolutionsLimit', 'weightPreset', 'disabledConstraints',
];

const DEFAULT_TIMEOUTS = {
  dwave: { stage1: 600, stage2: 1800 },
  scip:  { stage1: 1800, stage2: 1800 },
};

export function isValidRunId(id) {
  return typeof id === 'string' && RUN_ID_RE.test(id);
}

const present = (v) => v !== undefined && v !== null && String(v).trim() !== '';

function parseIntField(name, raw, { min, max }) {
  const s = String(raw).trim();
  if (!/^\d+$/.test(s)) return { error: `${name} must be a whole number` };
  const n = Number(s);
  if (n < min || (max != null && n > max)) {
    return { error: `${name} must be between ${min} and ${max ?? 'infinity'}` };
  }
  return { value: n };
}

function parseDisabledConstraints(raw) {
  let arr = raw;
  if (typeof raw === 'string') {
    try { arr = JSON.parse(raw); } catch { return { error: 'disabledConstraints must be a JSON array' }; }
  }
  if (!Array.isArray(arr)) return { error: 'disabledConstraints must be a JSON array' };
  if (new Set(arr).size !== arr.length) return { error: 'disabledConstraints must not contain duplicates' };
  const bad = arr.filter((c) => !DISABLED_CONSTRAINT_IDS.includes(c));
  if (bad.length) return { error: `disabledConstraints has unsupported value(s): ${bad.join(', ')}` };
  return { value: arr };
}

/**
 * Validate the text fields of POST /runSolver.
 *
 * @param body  req.body (multer text fields)
 * @param cfg   { sigmaiEnabled, dwaveEnabled, maxStageTimeoutSec }
 * @returns { ok: true, value } | { ok: false, error }
 *   value = { engine, option, label, originalEnvPath, originalSchedPath, solver | null }
 */
export function parseRunOptions(body = {}, cfg) {
  const engine = present(body.engine) ? String(body.engine).trim() : 'timefold';
  if (engine !== 'timefold' && engine !== 'sigmai') {
    return { ok: false, error: `engine must be "timefold" or "sigmai" (got "${engine}")` };
  }

  const label = present(body.label) ? String(body.label).replace(/[\r\n]+/g, ' ').trim() : '';
  if (label.length > 200) return { ok: false, error: 'label must be 200 characters or fewer' };

  const common = {
    engine,
    label,
    originalEnvPath: present(body.originalEnvPath) ? String(body.originalEnvPath) : null,
    originalSchedPath: present(body.originalSchedPath) ? String(body.originalSchedPath) : null,
  };

  if (engine === 'timefold') {
    const stray = SIGMAI_FIELDS.filter((f) => present(body[f]));
    if (stray.length) {
      return { ok: false, error: `Sigma-I field(s) not allowed with engine=timefold: ${stray.join(', ')}` };
    }
    return { ok: true, value: { ...common, option: 'timefold', solver: null } };
  }

  // ── engine = sigmai ──
  if (!cfg.sigmaiEnabled) {
    return { ok: false, error: 'Sigma-I is not enabled on this API controller (BATCH_JOB_ID_SIGMAI is not set)' };
  }
  const stage1Solver = present(body.stage1Solver) ? String(body.stage1Solver).trim() : 'scip';
  const stage2Solver = present(body.stage2Solver) ? String(body.stage2Solver).trim() : 'scip';
  if (stage1Solver !== 'scip' && stage1Solver !== 'dwave') {
    return { ok: false, error: `stage1Solver must be "scip" or "dwave" (got "${stage1Solver}")` };
  }
  if (stage2Solver !== 'scip') {
    return { ok: false, error: 'stage2Solver must be "scip" (Sigma-I only supports SCIP in Stage 2)' };
  }
  if (stage1Solver === 'dwave' && !cfg.dwaveEnabled) {
    return { ok: false, error: 'D-Wave is not enabled on this API controller (SIGMAI_DWAVE_ENABLED=false)' };
  }

  const range = { min: 1, max: cfg.maxStageTimeoutSec };
  const t1 = present(body.stage1TimeoutSec)
    ? parseIntField('stage1TimeoutSec', body.stage1TimeoutSec, range)
    : { value: DEFAULT_TIMEOUTS[stage1Solver].stage1 };
  if (t1.error) return { ok: false, error: t1.error };
  const t2 = present(body.stage2TimeoutSec)
    ? parseIntField('stage2TimeoutSec', body.stage2TimeoutSec, range)
    : { value: DEFAULT_TIMEOUTS[stage1Solver].stage2 };
  if (t2.error) return { ok: false, error: t2.error };

  const solver = {
    stage1: stage1Solver,
    stage2: stage2Solver,
    stage1TimeoutSec: t1.value,
    stage2TimeoutSec: t2.value,
    stage1NumSolutionsLimit: null,
    stage2NumSolutionsLimit: null,
    weightPreset: null,
    disabledConstraints: [],
  };

  for (const f of ['stage1NumSolutionsLimit', 'stage2NumSolutionsLimit']) {
    if (!present(body[f])) continue;
    const r = parseIntField(f, body[f], { min: 1, max: null });
    if (r.error) return { ok: false, error: r.error };
    solver[f] = r.value;
  }
  if (present(body.weightPreset)) {
    const w = String(body.weightPreset).trim();
    if (!WEIGHT_PRESETS.includes(w)) {
      return { ok: false, error: `weightPreset must be one of ${WEIGHT_PRESETS.join(', ')}` };
    }
    solver.weightPreset = w;
  }
  if (present(body.disabledConstraints)) {
    const d = parseDisabledConstraints(body.disabledConstraints);
    if (d.error) return { ok: false, error: d.error };
    solver.disabledConstraints = d.value;
  }

  return {
    ok: true,
    value: { ...common, option: `sigmai-${stage1Solver}-${stage2Solver}`, solver },
  };
}

/** Content of input/{runId}/sigmai_options.json — read by bridge.sh (see sigmai-setup/bridge.sh). */
export function buildSigmaiOptionsFile(parsed) {
  return {
    engine: 'sigmai',
    stage1Solver: parsed.solver.stage1,
    stage2Solver: parsed.solver.stage2,
    stage1TimeoutSec: parsed.solver.stage1TimeoutSec,
    stage2TimeoutSec: parsed.solver.stage2TimeoutSec,
    stage1NumSolutionsLimit: parsed.solver.stage1NumSolutionsLimit,
    stage2NumSolutionsLimit: parsed.solver.stage2NumSolutionsLimit,
    weightPreset: parsed.solver.weightPreset,
    disabledConstraints: parsed.solver.disabledConstraints,
    scenarioName: parsed.label || null,
  };
}

/** Batch maxWallClockTime (ISO 8601 duration) = both stage limits + margin. */
export function wallClockDuration(solver, marginSec) {
  const total = solver.stage1TimeoutSec + solver.stage2TimeoutSec + marginSec;
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `PT${h}H${m}M${s}S`;
}

/** 1 = next to run. `active` = [{id, creationTime}], oldest first or not. */
export function queuePositionFor(runId, thisCreationTime, activeTasks) {
  const mine = new Date(thisCreationTime).getTime();
  const ahead = activeTasks.filter(
    (t) => t.id !== runId && new Date(t.creationTime).getTime() <= mine,
  ).length;
  return ahead + 1;
}

export function buildInitialStatus({ runId, parsed, nowIso }) {
  return {
    runId, status: 'Submitted', stage: null, progress: 0,
    startedAt: nowIso, updatedAt: nowIso, finishedAt: null, error: null, output: null,
    engine: parsed.engine, option: parsed.option, solver: parsed.solver,
  };
}

export function buildManifest({ runId, parsed, inputInfo, versions, batchJobId, nowIso }) {
  return {
    schemaVersion: 1,
    runId,
    label: parsed.label || null,
    engine: parsed.engine,
    option: parsed.option,
    solver: parsed.solver,
    input: inputInfo,
    versions,
    submittedAt: nowIso,
    submittedBy: null,
    execution: { batchJobId, taskId: runId },
    result: null,
    deleted: false,
  };
}

/**
 * Fill execution + result into a manifest once the run is terminal. Idempotent: returns the
 * manifest unchanged when `result` is already set. `task` = Batch task JSON (may be null),
 * `sigmaiResult` = output/{runId}/sigmai/result.json (may be null).
 */
export function finalizeManifest(manifest, status, task, sigmaiResult) {
  if (manifest.result) return manifest;
  const info = task?.executionInfo ?? {};
  const start = info.startTime ? new Date(info.startTime).getTime() : null;
  const end = info.endTime ? new Date(info.endTime).getTime() : null;
  const created = task?.creationTime ? new Date(task.creationTime).getTime() : null;
  const sec = (a, b) => (a != null && b != null ? Math.max(0, Math.round((b - a) / 1000)) : null);

  const result = {
    status: status.status,
    error: status.error ?? null,
    output: status.status === 'Completed' ? (status.output ?? `output/${manifest.runId}/result_Schedule.yaml`) : null,
  };
  if (manifest.engine === 'sigmai') {
    const sm = sigmaiResult?.summary ?? status.sigmai?.summary ?? {};
    result.sigmai = {
      validationPassed: sm.validation_passed ?? sm.validationPassed ?? null,
      finalScore: sm.final_score ?? sm.finalScore ?? null,
      stage1Status: sigmaiResult?.stage1_result?.status ?? sm.stage1Status ?? null,
      stage2Status: sigmaiResult?.stage2_result?.status ?? sm.stage2Status ?? null,
      stage1SolveSec: sm.stage1_solve_time_sec ?? null,
      stage2SolveSec: sm.stage2_solve_time_sec ?? null,
      kpi: sigmaiResult?.kpi ?? null,
    };
  }
  return {
    ...manifest,
    execution: {
      ...manifest.execution,
      poolId: task?.nodeInfo?.poolId ?? null,
      nodeId: task?.nodeInfo?.nodeId ?? status.nodeId ?? null,
      queuedSec: sec(created, start),
      startedAt: info.startTime ?? null,
      finishedAt: info.endTime ?? status.finishedAt ?? null,
      runSec: sec(start, end),
      attempts: (info.retryCount ?? 0) + 1,
    },
    result,
  };
}

/**
 * Batch task already `completed` but the blob still shows a non-terminal status. Batch
 * uploads outputFiles right after completion, so allow a grace period before giving up.
 * Returns a synthesized Failed status, or null while still within the grace period.
 */
export function bridgeExitedWithoutStatus(status, task, graceSec, nowMs = Date.now()) {
  const end = task?.executionInfo?.endTime ? new Date(task.executionInfo.endTime).getTime() : null;
  if (end == null || nowMs - end < graceSec * 1000) return null;
  const nowIso = new Date(nowMs).toISOString();
  return {
    ...status,
    status: 'Failed',
    updatedAt: nowIso,
    finishedAt: nowIso,
    error: {
      type: 'BridgeExitedWithoutStatus',
      message: `The Batch task finished (exit code ${task.executionInfo?.exitCode ?? '?'}) but no final status reached Blob Storage.`,
    },
  };
}
