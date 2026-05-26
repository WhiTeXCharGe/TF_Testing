/**
 * Run store — the run log is derived from the REAL folders under
 * webapp/public/local/<id>/ (each folder = one run = one row).
 *
 *   public/local/<id>/input/EnvConfig.yaml + Schedule.yaml   → the run's input
 *   public/local/<id>/output/*.yaml                          → the solver result
 *
 * Folder discovery uses Vite's import.meta.glob (filesystem-based at dev/build
 * time), so adding a folder under public/local/ automatically adds a row.
 *
 * New Run uploads go through the /api/upload Vite dev middleware which writes
 * the files to public/local/<id>/input/ on disk. We also keep a localStorage
 * overlay for: original/saved paths, deleted ids, and fetched flags — anything
 * that isn't derivable from the on-disk folder alone.
 */
import type { Run } from '@/types';

// ── Discover real run folders under public/local/ ───────────────────────────
const INPUT_FILES  = import.meta.glob('/public/local/*/input/*.yaml');
const OUTPUT_FILES = import.meta.glob('/public/local/*/output/*.yaml');

function idsFrom(glob: Record<string, unknown>, seg: 'input' | 'output'): Set<string> {
  const re = new RegExp(`/local/([^/]+)/${seg}/`);
  const ids = new Set<string>();
  for (const key of Object.keys(glob)) {
    const m = key.match(re);
    if (m) ids.add(m[1]);
  }
  return ids;
}

/** Find the first matching path under public/local/<id>/<seg>/ for the glob result. */
function firstPath(glob: Record<string, unknown>, id: string, seg: 'input' | 'output', namePattern?: RegExp): string | undefined {
  const prefix = `/public/local/${id}/${seg}/`;
  for (const key of Object.keys(glob)) {
    if (key.startsWith(prefix)) {
      const name = key.slice(prefix.length);
      if (!namePattern || namePattern.test(name)) {
        // Strip the leading /public so the path matches what the browser sees.
        return key.replace('/public', '');
      }
    }
  }
  return undefined;
}

/** Derive an ISO solve date from a folder id like "20260521" → 2026-05-21. */
function dateFromId(id: string): string {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}T09:00:00`;
  return new Date().toISOString();
}

function folderRuns(): Run[] {
  const inputIds  = idsFrom(INPUT_FILES, 'input');
  const outputIds = idsFrom(OUTPUT_FILES, 'output');
  return [...inputIds].map((id): Run => {
    const hasOutput = outputIds.has(id);
    const envPath   = firstPath(INPUT_FILES, id, 'input', /env/i);
    const schedPath = firstPath(INPUT_FILES, id, 'input', /sched/i);
    const outPath   = firstPath(OUTPUT_FILES, id, 'output');
    return {
      id,
      solveDate: dateFromId(id),
      label: 'Local run',
      folderPath: `./local/${id}/`,
      inputEnvName: envPath ? envPath.split('/').pop()! : 'EnvConfig.yaml',
      inputSchedName: schedPath ? schedPath.split('/').pop()! : 'Schedule.yaml',
      inputDir: `/local/${id}/input`,
      output: hasOutput ? 'ready' : 'none',
      outputHasYaml: hasOutput,
      savedEnvPath: envPath,
      savedSchedPath: schedPath,
      savedOutputPath: outPath,
    };
  });
}

// ── localStorage overlay (path metadata, deleted ids, fetched flags) ────────
const RUNS_KEY    = 'tfScheduler_userRuns';
const FETCHED_KEY = 'tfScheduler_fetched';
const DELETED_KEY = 'tfScheduler_deleted';
const META_KEY    = 'tfScheduler_meta';

type RunMeta = Partial<Pick<Run, 'originalEnvPath' | 'originalSchedPath' | 'savedEnvPath' | 'savedSchedPath'>>;

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw) return JSON.parse(raw) as T;
  } catch { /* ignore */ }
  return fallback;
}
function writeJSON(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

function readUserRuns(): Run[]      { return readJSON<Run[]>(RUNS_KEY, []); }
function writeUserRuns(r: Run[]): void { writeJSON(RUNS_KEY, r); }
function readFetched(): Set<string> { return new Set(readJSON<string[]>(FETCHED_KEY, [])); }
function writeFetched(s: Set<string>): void { writeJSON(FETCHED_KEY, [...s]); }
function readDeleted(): Set<string> { return new Set(readJSON<string[]>(DELETED_KEY, [])); }
function writeDeleted(s: Set<string>): void { writeJSON(DELETED_KEY, [...s]); }
function readMeta(): Record<string, RunMeta> { return readJSON<Record<string, RunMeta>>(META_KEY, {}); }
function writeMeta(m: Record<string, RunMeta>): void { writeJSON(META_KEY, m); }

// ── Public API ──────────────────────────────────────────────────────────────

/** All runs (real folders + user-created), newest first, fetched-flags applied. */
export function listRuns(): Run[] {
  const fetched = readFetched();
  const deleted = readDeleted();
  const meta    = readMeta();
  const byId = new Map<string, Run>();

  for (const r of folderRuns()) {
    if (deleted.has(r.id)) continue;
    byId.set(r.id, r);
  }
  for (const r of readUserRuns()) {
    if (deleted.has(r.id)) continue;
    byId.set(r.id, r);   // user rows can extend the list
  }

  const runs = [...byId.values()].map(r => {
    let out = r;
    if (fetched.has(r.id) && r.output !== 'ready') {
      out = { ...out, output: 'ready' as const };
    }
    const m = meta[r.id];
    if (m) {
      out = { ...out, ...m };
    }
    return out;
  });

  return runs.sort((a, b) => b.solveDate.localeCompare(a.solveDate));
}

export function getRun(id: string): Run | undefined {
  return listRuns().find(r => r.id === id);
}

interface CreateRunArgs {
  envName: string;
  schedName: string;
  label?: string;
  inputDir?: string | null;
  originalEnvPath?: string;
  originalSchedPath?: string;
  savedEnvPath?: string;
  savedSchedPath?: string;
}

/** Generate a unique run id like 20260527_001 that doesn't clash with existing folders. */
export function nextRunId(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  const base = `${y}${m}${d}`;
  const existingIds = new Set(listRuns().map(r => r.id));
  for (let i = 1; i < 1000; i++) {
    const id = `${base}_${String(i).padStart(3, '0')}`;
    if (!existingIds.has(id)) return id;
  }
  return `${base}_${Date.now()}`;
}

/** Create a user run row. Caller is responsible for uploading files first. */
export function createRun(args: CreateRunArgs & { id?: string }): Run {
  const id = args.id ?? nextRunId();
  const now = new Date();

  const run: Run = {
    id,
    solveDate: now.toISOString(),
    label: args.label?.trim() || 'New run',
    folderPath: `./local/${id}/`,
    inputEnvName: args.envName,
    inputSchedName: args.schedName,
    inputDir: args.inputDir ?? `/local/${id}/input`,
    output: 'none',
    outputHasYaml: false,
    originalEnvPath:   args.originalEnvPath,
    originalSchedPath: args.originalSchedPath,
    savedEnvPath:      args.savedEnvPath,
    savedSchedPath:    args.savedSchedPath,
  };

  // Remove any prior deleted-flag so the row re-appears if id is reused.
  const del = readDeleted(); del.delete(id); writeDeleted(del);

  writeUserRuns([run, ...readUserRuns().filter(r => r.id !== id)]);
  return run;
}

/** Mark a run's output/ as fetched. */
export function markFetched(id: string): void {
  const f = readFetched();
  f.add(id);
  writeFetched(f);
}

/** Persist extra metadata (paths) for a run id without changing the run row. */
export function setRunMeta(id: string, m: RunMeta): void {
  const all = readMeta();
  all[id] = { ...(all[id] || {}), ...m };
  writeMeta(all);
}

/** Soft-delete a run: removes from overlay + marks the on-disk id as hidden. */
export function deleteRun(id: string): void {
  writeUserRuns(readUserRuns().filter(r => r.id !== id));
  const f = readFetched(); f.delete(id); writeFetched(f);
  const m = readMeta(); delete m[id]; writeMeta(m);
  const d = readDeleted(); d.add(id); writeDeleted(d);
}

/** Clear the entire localStorage overlay. */
export function resetRuns(): void {
  try {
    localStorage.removeItem(RUNS_KEY);
    localStorage.removeItem(FETCHED_KEY);
    localStorage.removeItem(DELETED_KEY);
    localStorage.removeItem(META_KEY);
  } catch { /* ignore */ }
}
