import { useCallback, useState } from 'react';
import type { Run } from '@/types';
import {
  listRuns, createRun, markFetched, deleteRun, nextRunId, setRunMeta,
} from '@/services/runStore';

export interface NewRunPayload {
  envFile: File;
  schedFile: File;
  originalEnvPath: string;
  originalSchedPath: string;
  label?: string;
}

interface UseRuns {
  runs: Run[];
  refresh: () => void;
  /**
   * Upload the two input files to /api/upload (Vite dev middleware), then
   * create a new run row. Returns the created Run on success.
   */
  submitNewRun: (payload: NewRunPayload) => Promise<Run>;
  /** Mark a run's output as fetched (mock; no real download). */
  fetchOutput: (id: string) => Promise<void>;
  /** Check on disk whether public/local/<id>/output has a yaml. */
  checkOutput: (id: string) => Promise<{ hasYaml: boolean; yamlPath: string | null }>;
  /** Delete a run row and its public/local/<id>/ folder. */
  removeRun: (id: string) => Promise<void>;
}

export function useRuns(): UseRuns {
  const [runs, setRuns] = useState<Run[]>(() => listRuns());

  const refresh = useCallback(() => setRuns(listRuns()), []);

  const submitNewRun = useCallback(async (p: NewRunPayload): Promise<Run> => {
    const id = nextRunId();
    const form = new FormData();
    form.append('runId', id);
    form.append('env',   p.envFile,   p.envFile.name);
    form.append('sched', p.schedFile, p.schedFile.name);

    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`Upload failed (${res.status}): ${body || res.statusText}`);
    }
    const data: { runId: string; saved: Record<string, string> } = await res.json();

    const run = createRun({
      id: data.runId,
      envName: p.envFile.name,
      schedName: p.schedFile.name,
      label: p.label,
      originalEnvPath:   p.originalEnvPath.trim() || undefined,
      originalSchedPath: p.originalSchedPath.trim() || undefined,
      savedEnvPath:      data.saved.env,
      savedSchedPath:    data.saved.sched,
    });
    setRunMeta(run.id, {
      originalEnvPath:   run.originalEnvPath,
      originalSchedPath: run.originalSchedPath,
      savedEnvPath:      run.savedEnvPath,
      savedSchedPath:    run.savedSchedPath,
    });
    setRuns(listRuns());
    return run;
  }, []);

  const fetchOutput = useCallback(async (id: string) => {
    await new Promise(res => setTimeout(res, 800));
    markFetched(id);
    setRuns(listRuns());
  }, []);

  const checkOutput = useCallback(async (id: string) => {
    try {
      const res = await fetch(`/api/run/${encodeURIComponent(id)}/output`);
      if (!res.ok) return { hasYaml: false, yamlPath: null };
      return await res.json() as { hasYaml: boolean; yamlPath: string | null };
    } catch {
      return { hasYaml: false, yamlPath: null };
    }
  }, []);

  const removeRun = useCallback(async (id: string) => {
    // Try to delete the folder; even if the API call fails (e.g. folder
    // doesn't exist on disk), still soft-delete the localStorage row.
    try {
      await fetch(`/api/run/${encodeURIComponent(id)}`, { method: 'DELETE' });
    } catch { /* ignore */ }
    deleteRun(id);
    setRuns(listRuns());
  }, []);

  return { runs, refresh, submitNewRun, fetchOutput, checkOutput, removeRun };
}
