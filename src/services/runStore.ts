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
 * A browser cannot create folders on disk, so "New Run" / "Fetch Result" are
 * tracked as lightweight overrides in localStorage layered on top of the real
 * folder list. To make them real later, swap this module for the File System
 * Access API or a small local Node backend — the UI won't change.
 */
import type { Run } from '@/types';

// ── Discover real run folders under public/local/ ───────────────────────────
const INPUT_FILES  = import.meta.glob('/public/local/*/input/EnvConfig.yaml');
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
    return {
      id,
      solveDate: dateFromId(id),
      label: 'Local run',
      folderPath: `./local/${id}/`,
      inputEnvName: 'EnvConfig.yaml',
      inputSchedName: 'Schedule.yaml',
      inputDir: `/local/${id}/input`,
      output: hasOutput ? 'ready' : 'none',
      outputHasYaml: hasOutput,
    };
  });
}

// ── localStorage overlay (New Run rows + fetched flags) ─────────────────────
const RUNS_KEY    = 'tfScheduler_userRuns';
const FETCHED_KEY = 'tfScheduler_fetched';

function readUserRuns(): Run[] {
  try {
    const raw = localStorage.getItem(RUNS_KEY);
    if (raw) return JSON.parse(raw) as Run[];
  } catch { /* ignore */ }
  return [];
}
function writeUserRuns(runs: Run[]): void {
  try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs)); } catch { /* ignore */ }
}
function readFetched(): Set<string> {
  try {
    const raw = localStorage.getItem(FETCHED_KEY);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch { /* ignore */ }
  return new Set();
}
function writeFetched(ids: Set<string>): void {
  try { localStorage.setItem(FETCHED_KEY, JSON.stringify([...ids])); } catch { /* ignore */ }
}

// ── Public API ──────────────────────────────────────────────────────────────

/** All runs (real folders + user-created), newest first, fetched-flags applied. */
export function listRuns(): Run[] {
  const fetched = readFetched();
  const byId = new Map<string, Run>();

  for (const r of folderRuns()) byId.set(r.id, r);
  for (const r of readUserRuns()) byId.set(r.id, r);   // user rows can extend the list

  const runs = [...byId.values()].map(r => {
    // A fetched flag marks the output/ folder as populated.
    if (fetched.has(r.id) && r.output !== 'ready') return { ...r, output: 'ready' as const };
    return r;
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
}

/** Create a user run row (localStorage overlay; no real folder is written). */
export function createRun({ envName, schedName, label, inputDir = null }: CreateRunArgs): Run {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  const existing = listRuns().length;
  const id = `${y}${m}${d}_${String(existing + 1).padStart(3, '0')}`;

  const run: Run = {
    id,
    solveDate: now.toISOString(),
    label: label?.trim() || 'New run',
    folderPath: `./local/${id}/`,
    inputEnvName: envName,
    inputSchedName: schedName,
    inputDir,
    output: 'none',
    outputHasYaml: false,
  };

  writeUserRuns([run, ...readUserRuns()]);
  return run;
}

/** Mark a run's output/ as fetched (does not enable the result Gantt by itself). */
export function markFetched(id: string): void {
  const f = readFetched();
  f.add(id);
  writeFetched(f);
}

/** Clear the localStorage overlay (user rows + fetched flags). */
export function resetRuns(): void {
  try {
    localStorage.removeItem(RUNS_KEY);
    localStorage.removeItem(FETCHED_KEY);
  } catch { /* ignore */ }
}
