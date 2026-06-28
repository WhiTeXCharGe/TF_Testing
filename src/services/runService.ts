/**
 * Solver API client — talks to the backend service (local Express or Azure ACA).
 * Set VITE_API_BASE_URL in webapp/.env to enable; leave blank for local-only mode.
 */
import axios, { AxiosError } from 'axios';
import { APP_CONFIG } from '@/config/appConfig';

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
  form.append('env',   envFile,   envFile.name);
  form.append('sched', schedFile, schedFile.name);

  try {
    const res = await axios.post<SubmitResult>(
      `${APP_CONFIG.apiBaseUrl}/runSolver`,
      form,
    );
    return res.data;
  } catch (err) {
    throw new Error(extractMessage(err, 'Upload to solver failed'));
  }
}

/**
 * GET /status/:runId
 * Returns the current solve status. Throws a human-readable Error on failure.
 */
export async function checkStatus(runId: string): Promise<RunStatus> {
  try {
    const res = await axios.get<RunStatus>(
      `${APP_CONFIG.apiBaseUrl}/status/${encodeURIComponent(runId)}`,
    );
    return res.data;
  } catch (err) {
    throw new Error(extractMessage(err, 'Failed to get run status'));
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
      `${APP_CONFIG.apiBaseUrl}/download/${encodeURIComponent(runId)}`,
      { responseType: 'blob' },
    );

    // Extract filename from Content-Disposition if the server sends it.
    const disposition: string = (res.headers['content-disposition'] as string) ?? '';
    const match = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
    const filename = match?.[1]?.replace(/['"]/g, '') || 'result_Schedule.yaml';

    return { blob: res.data as Blob, filename };
  } catch (err) {
    throw new Error(extractMessage(err, 'Download failed'));
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
      `${APP_CONFIG.apiBaseUrl}/run/${encodeURIComponent(runId)}`,
    );
  } catch (err) {
    throw new Error(extractMessage(err, 'Cancel failed'));
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