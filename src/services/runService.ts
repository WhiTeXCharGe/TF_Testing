/**
 * Solver API client — talks to the backend service (local Express or Azure ACA).
 * The base URL comes from config.txt at runtime (see services/solverConfig.ts).
 */
import axios, { AxiosError } from 'axios';
import { APP_CONFIG } from '@/config/appConfig';
import { getSolverBaseUrl } from '@/services/solverConfig';
import { UI } from '@/config/uiConfig';
import {
  getSolverOption, toRequestFields,
  type SolverEngine, type SolverSelection,
} from '@/services/solverOptions';

export interface SubmitResult {
  runId: string;
  status: string;
}

export interface RunError {
  type?: string;
  message?: string;
}

export interface RunStatus {
  runId?: string;
  status: 'Submitted' | 'Running' | 'Completed' | 'Failed' | 'Cancelled';
  stage: number | string | null;
  progress?: number;
  startedAt?: string | null;
  updatedAt?: string | null;
  finishedAt?: string | null;
  error?: string | RunError | null;
  output?: string | null;
  /** Set by the v2 API controller. Absent on the Timefold-only controller. */
  engine?: SolverEngine;
  /** Only while waiting in the Batch queue (Sigma-I pool). 1 = next to run. */
  queuePosition?: number;
}

/**
 * POST /runSolver
 * Upload EnvConfig + Schedule YAMLs to the solver backend.
 * Pass the same runId used for the local upload so both sides stay in sync.
 */
export async function submitRun(
  runId:     string,
  envFile:   File,
  schedFile: File,
  selection?: SolverSelection,
): Promise<SubmitResult> {
  const form = new FormData();
  form.append('runId', runId);
  // Timefold-only backends ignore unknown fields, so a Sigma-I run sent to one would
  // silently run on Timefold. ensureEngineSupported() refuses that case up front.
  if (selection) {
    await ensureEngineSupported(getSolverOption(selection.option).engine);
    for (const [k, v] of Object.entries(toRequestFields(selection))) form.append(k, v);
  }
  // The solver stores uploads under their filename and Azure Batch looks for
  // exactly EnvConfig.yaml / Schedule.yaml, so always send canonical names.
  form.append('env',   envFile,   APP_CONFIG.envConfigFile);
  form.append('sched', schedFile, APP_CONFIG.scheduleFile);

  try {
    const res = await axios.post<SubmitResult>(
      `${await getSolverBaseUrl()}/runSolver`,
      form,
    );
    return res.data;
  } catch (err) {
    throw new Error(extractMessage(err, UI.errors.solverUploadFailed));
  }
}

/**
 * GET /health → { ok, engines?: string[] }
 * The Timefold-only controller returns { ok: true } (no `engines`) → only 'timefold'.
 * The v2 controller lists the engines it can route. Throws if `engine` is not available.
 */
export async function ensureEngineSupported(engine: SolverEngine): Promise<void> {
  if (engine === 'timefold') return;
  let engines: string[] = ['timefold'];
  try {
    const res = await axios.get<{ engines?: string[] }>(`${await getSolverBaseUrl()}/health`);
    if (Array.isArray(res.data?.engines)) engines = res.data.engines;
  } catch (err) {
    throw new Error(extractMessage(err, UI.errors.solverUploadFailed));
  }
  if (!engines.includes(engine)) throw new Error(UI.errors.engineNotSupported(engine));
}

/**
 * GET /status/:runId
 * Returns the current solve status. Throws a human-readable Error on failure.
 */
export async function checkStatus(runId: string): Promise<RunStatus> {
  try {
    const res = await axios.get<RunStatus>(
      `${await getSolverBaseUrl()}/status/${encodeURIComponent(runId)}`,
    );
    return res.data;
  } catch (err) {
    throw new Error(extractMessage(err, UI.errors.solverStatusFailed));
  }
}

/**
 * GET /download/:runId
 * Downloads the output YAML as a Blob. Only call when status === 'Completed'.
 * Returns { blob, filename } — pass both to file-saver's saveAs().
 */
export async function downloadOutput(
  runId: string,
): Promise<{ blob: Blob; filename: string }> {
  try {
    const res = await axios.get(
      `${await getSolverBaseUrl()}/download/${encodeURIComponent(runId)}`,
      { responseType: 'blob' },
    );

    // Extract filename from Content-Disposition if the server sends it.
    const disposition: string = (res.headers['content-disposition'] as string) ?? '';
    const match = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
    const filename = match?.[1]?.replace(/['"]/g, '') || 'result_Schedule.yaml';

    return { blob: res.data as Blob, filename };
  } catch (err) {
    throw new Error(extractMessage(err, UI.errors.solverDownloadFailed));
  }
}

/**
 * DELETE /run/:runId
 * Cancel a running container and delete all service-side data for the run.
 * Non-fatal if the run was never submitted to the solver.
 */
export async function cancelRun(runId: string): Promise<void> {
  try {
    await axios.delete(
      `${await getSolverBaseUrl()}/run/${encodeURIComponent(runId)}`,
    );
  } catch (err) {
    throw new Error(extractMessage(err, UI.errors.solverCancelFailed));
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function extractMessage(err: unknown, fallback: string): string {
  if (err instanceof AxiosError) {
    // Try to read a JSON { error: "..." } body from the server.
    const data = err.response?.data;
    if (data && typeof data === 'object' && 'error' in data) {
      return String((data as Record<string, unknown>).error);
    }
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return fallback;
}