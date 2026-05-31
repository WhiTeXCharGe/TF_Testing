import { useCallback, useEffect, useState } from 'react';
import type { Run } from '@/types';
import {
  fetchRuns, uploadRun, deleteRun as apiDeleteRun, checkOutput,
  type UploadPayload,
} from '@/services/runStore';

interface UseRuns {
  runs: Run[];
  loading: boolean;
  error: string | null;
  /** Re-fetch the runs.json database. */
  refresh: () => Promise<void>;
  /** Upload files + persist to runs.json. Returns the newly-created Run. */
  submitNewRun: (payload: UploadPayload) => Promise<Run>;
  /** Check whether public/local/<id>/output/ has a yaml. */
  checkOutput: (id: string) => Promise<{ hasYaml: boolean; yamlPath: string | null }>;
  /** Delete from runs.json + remove public/local/<id>/ on disk. */
  removeRun: (id: string) => Promise<void>;
}

export function useRuns(): UseRuns {
  const [runs, setRuns]       = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

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
    // The middleware also writes savedOutputPath back to runs.json, so a
    // refresh picks up the new path for the result-cell box.
    if (out.hasYaml) await refresh();
    return out;
  }, [refresh]);

  const removeRun = useCallback(async (id: string) => {
    await apiDeleteRun(id);
    await refresh();
  }, [refresh]);

  return {
    runs, loading, error, refresh,
    submitNewRun,
    checkOutput: checkOutputWrapped,
    removeRun,
  };
}
