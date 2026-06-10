import { useCallback, useEffect, useState } from 'react';
import type { Run } from '@/types';
import { APP_CONFIG } from '@/config/appConfig';
import {
  fetchRuns, uploadRun, deleteRun as apiDeleteRun, checkOutput,
  type UploadPayload,
} from '@/services/runStore';
import {
  submitRun, checkStatus, downloadOutput,
  type RunStatus,
} from '@/services/runService';

export interface UseRuns {
  runs:     Run[];
  loading:  boolean;
  error:    string | null;
  /** Re-fetch runs.json. */
  refresh: () => Promise<void>;
  /** Save files locally + show in run list. Returns the new Run row. */
  submitNewRun: (payload: UploadPayload) => Promise<Run>;
  /** Check whether public/local/<id>/output/ has a yaml (local-only mode). */
  checkOutput: (id: string) => Promise<{ hasYaml: boolean; yamlPath: string | null }>;
  /** Delete from runs.json + remove public/local/<id>/ on disk. */
  removeRun: (id: string) => Promise<void>;

  // ── Solver API (only active when VITE_API_BASE_URL is set) ─────────────────
  /** true when VITE_API_BASE_URL is configured — gates solver API calls. */
  solverEnabled: boolean;
  /** POST /runSolver — send the two YAMLs to the solver backend. */
  submitToSolver: (runId: string, envFile: File, schedFile: File) => Promise<void>;
  /** GET /status/:runId — get current solve status from the backend. */
  checkRunStatus: (runId: string) => Promise<RunStatus>;
  /** GET /download/:runId — download output YAML as a Blob. */
  triggerDownload: (runId: string) => Promise<{ blob: Blob; filename: string }>;
}

export function useRuns(): UseRuns {
  const [runs,    setRuns]    = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await fetchRuns();
      setRuns(list);
      setError(null);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const submitNewRun = useCallback(async (p: UploadPayload): Promise<Run> => {
    const run = await uploadRun(p);
    await refresh();
    return run;
  }, [refresh]);

  const checkOutputWrapped = useCallback(async (id: string) => {
    const out = await checkOutput(id);
    if (out.hasYaml) await refresh();
    return out;
  }, [refresh]);

  const removeRun = useCallback(async (id: string) => {
    await apiDeleteRun(id);
    await refresh();
  }, [refresh]);

  // ── Solver API ──────────────────────────────────────────────────────────────

  const submitToSolver = useCallback(async (
    runId:     string,
    envFile:   File,
    schedFile: File,
  ) => {
    await submitRun(runId, envFile, schedFile);
  }, []);

  const checkRunStatus = useCallback(
    (runId: string) => checkStatus(runId),
    [],
  );

  const triggerDownload = useCallback(
    (runId: string) => downloadOutput(runId),
    [],
  );

  return {
    runs, loading, error, refresh,
    submitNewRun,
    checkOutput: checkOutputWrapped,
    removeRun,
    solverEnabled: Boolean(APP_CONFIG.apiBaseUrl),
    submitToSolver,
    checkRunStatus,
    triggerDownload,
  };
}
