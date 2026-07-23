/**
 * Run store — API client for the JSON-backed run database.
 *
 * Source of truth: webapp/public/local/runs.json (managed by the Vite dev
 * middleware in vite.config.ts, or by a real backend in production).
 *
 * The browser NEVER scans public/local/ folders directly. A folder dropped
 * manually under public/local/<id>/ will NOT appear in the Run Log — only
 * entries present in runs.json are shown.
 */
import type { Run } from '@/types';

interface RunsResponse { runs: Run[] }
interface UploadResponse { run: Run }
interface OutputResponse { hasYaml: boolean; yamlPath: string | null }

const API = {
  runs:   '/api/runs',
  upload: '/api/upload',
  run:    (id: string) => `/api/run/${encodeURIComponent(id)}`,
  output: (id: string) => `/api/run/${encodeURIComponent(id)}/output`,
  saveOutput: (id: string, filename: string) =>
    `/api/run/${encodeURIComponent(id)}/output?filename=${encodeURIComponent(filename)}`,
};

/** Load all runs from the JSON database. Newest first. */
export async function fetchRuns(): Promise<Run[]> {
  const res = await fetch(API.runs, { cache: 'no-store' });
  if (!res.ok) throw new Error(`fetchRuns failed (${res.status})`);
  const data = await res.json() as RunsResponse;
  return (data.runs ?? []).sort((a, b) => b.solveDate.localeCompare(a.solveDate));
}

export interface UploadPayload {
  /** The actual File object (from a file picker or drag-drop). */
  envFile: File;
  /** The actual File object (from a file picker or drag-drop). */
  schedFile: File;
  /** User-typed original disk path for the EnvConfig file. Optional. */
  originalEnvPath?: string;
  /** User-typed original disk path for the Schedule file. Optional. */
  originalSchedPath?: string;
  label?: string;
}

/**
 * Upload the two input files via multipart. The middleware writes them to
 * public/local/<runId>/input/ and appends a row to runs.json. The original
 * paths (if provided) are persisted verbatim — the browser can't expose
 * real disk paths so the user types them.
 */
export async function uploadRun(p: UploadPayload): Promise<Run> {
  const id = nextRunIdFromTimestamp();
  const form = new FormData();
  form.append('runId', id);
  form.append('env',   p.envFile,   p.envFile.name);
  form.append('sched', p.schedFile, p.schedFile.name);
  if (p.originalEnvPath?.trim())   form.append('originalEnvPath',   p.originalEnvPath.trim());
  if (p.originalSchedPath?.trim()) form.append('originalSchedPath', p.originalSchedPath.trim());
  if (p.label?.trim())             form.append('label',             p.label.trim());

  const res = await fetch(API.upload, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Upload failed (${res.status}): ${body || res.statusText}`);
  }
  const data = await res.json() as UploadResponse;
  return data.run;
}

/** Delete a run row from the JSON database AND remove its folder on disk. */
export async function deleteRun(id: string): Promise<void> {
  const res = await fetch(API.run(id), { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Delete failed (${res.status}): ${body || res.statusText}`);
  }
}

/**
 * Check whether public/local/<id>/output/ contains a yaml. The middleware
 * also writes savedOutputPath back to runs.json when a yaml is found.
 */
export async function checkOutput(id: string): Promise<OutputResponse> {
  try {
    const res = await fetch(API.output(id), { cache: 'no-store' });
    if (!res.ok) return { hasYaml: false, yamlPath: null };
    return await res.json() as OutputResponse;
  } catch {
    return { hasYaml: false, yamlPath: null };
  }
}

/**
 * Persist a solver-output blob (downloaded from the Azure API) to
 * local/<id>/output/<filename> via the local API server. This is what
 * actually lands the file at the path shown in the UI — previously that path
 * was only ever a display string; the real bytes only reached the browser's
 * Downloads folder via saveAs().
 */
export async function saveOutput(id: string, blob: Blob, filename: string): Promise<string> {
  const res = await fetch(API.saveOutput(id, filename), {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: blob,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`出力の保存に失敗しました (${res.status}): ${body || res.statusText}`);
  }
  const data = await res.json() as { ok: boolean; yamlPath: string };
  return data.yamlPath;
}

/** Clear runs.json (folders on disk are NOT removed — clean them up manually if needed). */
export async function resetRuns(): Promise<void> {
  const res = await fetch(API.runs, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Reset failed (${res.status}): ${body || res.statusText}`);
  }
}

/** Generate a unique run id like 20260527_HHMMSSmmm. The server uses this as the folder name. */
function nextRunIdFromTimestamp(): string {
  const d = new Date();
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}${pad(d.getMilliseconds(), 3)}`
  );
}
