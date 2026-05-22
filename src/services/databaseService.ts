/**
 * Database service — reads from database.xlsx in /public/data/.
 * Falls back to mockData if the file is not found.
 * Uses localStorage to cache parsed data within a session.
 */
import * as XLSX from 'xlsx';
import { APP_CONFIG } from '@/config/appConfig';
import type { Dataset, RunLog, Comment } from '@/types';
import { MOCK_DATASETS, MOCK_RUN_LOGS, MOCK_COMMENTS } from '@/data/mockData';

interface DbCache {
  datasets: Dataset[];
  runLogs: RunLog[];
  comments: Comment[];
}

let memCache: DbCache | null = null;

/** Load and parse database.xlsx, with localStorage caching. */
export async function loadDatabase(): Promise<DbCache> {
  if (memCache) return memCache;

  // Check localStorage cache
  try {
    const ts = localStorage.getItem(APP_CONFIG.storage.dbCacheTs);
    if (ts && Date.now() - Number(ts) < APP_CONFIG.cacheTtlMs) {
      const raw = localStorage.getItem(APP_CONFIG.storage.dbCache);
      if (raw) {
        memCache = JSON.parse(raw) as DbCache;
        return memCache;
      }
    }
  } catch {
    // ignore localStorage errors
  }

  // Fetch from /public/data/database.xlsx
  try {
    const url = `${APP_CONFIG.dataBasePath}/${APP_CONFIG.databaseFileName}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    const wb = XLSX.read(buf, { type: 'array' });

    const datasets = parseSheet<Dataset>(wb, APP_CONFIG.sheets.datasets);
    const runLogs  = parseSheet<RunLog>(wb, APP_CONFIG.sheets.runLogs);
    const comments = parseSheet<Comment>(wb, APP_CONFIG.sheets.comments);

    memCache = { datasets, runLogs, comments };

    // Cache in localStorage
    try {
      localStorage.setItem(APP_CONFIG.storage.dbCache, JSON.stringify(memCache));
      localStorage.setItem(APP_CONFIG.storage.dbCacheTs, String(Date.now()));
    } catch { /* quota exceeded — ignore */ }

    return memCache;
  } catch {
    console.warn('database.xlsx not found — using mock data.');
    memCache = {
      datasets: MOCK_DATASETS,
      runLogs:  MOCK_RUN_LOGS,
      comments: MOCK_COMMENTS,
    };
    return memCache;
  }
}

/** Parse a worksheet into typed objects using header row as keys. */
function parseSheet<T>(wb: XLSX.WorkBook, sheetName: string): T[] {
  const ws = wb.Sheets[sheetName];
  if (!ws) return [];
  return XLSX.utils.sheet_to_json<T>(ws, { defval: null });
}

// ── Public accessors ──────────────────────────────────────────────

export async function getDatasets(): Promise<Dataset[]> {
  const db = await loadDatabase();
  return db.datasets;
}

export async function getRunLogs(datasetId: string): Promise<RunLog[]> {
  const db = await loadDatabase();
  return db.runLogs.filter(r => r.datasetId === datasetId);
}

export async function getComments(datasetId: string): Promise<Comment[]> {
  const db = await loadDatabase();
  return db.comments.filter(c => c.datasetId === datasetId);
}

/**
 * Add a new comment (session-only, stored in memCache + localStorage).
 * A proper backend would persist this to the DB.
 */
export async function addComment(comment: Comment): Promise<void> {
  const db = await loadDatabase();
  db.comments.push(comment);
  try {
    localStorage.setItem(APP_CONFIG.storage.dbCache, JSON.stringify(db));
    localStorage.setItem(APP_CONFIG.storage.dbCacheTs, String(Date.now()));
  } catch { /* ignore */ }
}

/** Clear cache (useful for dev / forced refresh). */
export function clearCache(): void {
  memCache = null;
  localStorage.removeItem(APP_CONFIG.storage.dbCache);
  localStorage.removeItem(APP_CONFIG.storage.dbCacheTs);
}
