import { useCallback, useState } from 'react';
import type { Run } from '@/types';
import { listRuns, createRun, markFetched } from '@/services/runStore';

interface UseRuns {
  runs: Run[];
  refresh: () => void;
  /** Create a new run (new folder + log row) and refresh the list. */
  addRun: (args: { envName: string; schedName: string; label?: string; inputDir?: string | null }) => Run;
  /** Simulate fetching the solver output into the run's output/ folder. */
  fetchOutput: (id: string) => Promise<void>;
}

export function useRuns(): UseRuns {
  const [runs, setRuns] = useState<Run[]>(() => listRuns());

  const refresh = useCallback(() => setRuns(listRuns()), []);

  const addRun: UseRuns['addRun'] = useCallback((args) => {
    const run = createRun(args);
    setRuns(listRuns());
    return run;
  }, []);

  const fetchOutput = useCallback(async (id: string) => {
    // Simulate a ~1.8s cloud download into the local output/ folder.
    await new Promise(res => setTimeout(res, 1800));
    markFetched(id);
    setRuns(listRuns());
  }, []);

  return { runs, refresh, addRun, fetchOutput };
}
