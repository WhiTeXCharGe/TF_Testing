/**
 * Solver API client — talks to the backend service (local Express or Azure ACA).
 * The base URL comes from config.txt at runtime (see services/solverConfig.ts).
 */
import axios, { AxiosError } from 'axios';
import { APP_CONFIG } from '@/config/appConfig';
import { getSolverBaseUrl } from '@/services/solverConfig';
import { UI } from '@/config/uiConfig';

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
): Promise<SubmitResult> {
  const form = new FormData();
  form.append('runId', runId);
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