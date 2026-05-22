import { useState, useEffect } from 'react';
import type { GanttData } from '@/types';
import { loadEnvConfig, loadSchedule } from '@/utils/yamlUtils';
import { buildGanttData } from '@/services/ganttService';
import { MOCK_ENV_CONFIG, MOCK_SCHEDULE } from '@/data/mockData';

interface UseGantt {
  ganttData: GanttData | null;
  loading: boolean;
  error: string | null;
  /** true when the Gantt was built from the mock fallback (no real YAML). */
  isMock: boolean;
}

/**
 * Build Gantt data for a run.
 *  - inputDir set  → fetch the real EnvConfig.yaml + Schedule.yaml from that
 *                    public folder and port them (yaml_to_suother_like_excel.py).
 *  - inputDir null → mock fallback: feed MOCK_* through the SAME builder.
 */
export function useGantt(inputDir: string | null): UseGantt {
  const [ganttData, setGanttData] = useState<GanttData | null>(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [isMock, setIsMock]       = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    // ── Mock fallback ──────────────────────────────────────────
    if (!inputDir) {
      try {
        const data = buildGanttData(MOCK_ENV_CONFIG, MOCK_SCHEDULE);
        if (!cancelled) { setGanttData(data); setIsMock(true); setLoading(false); }
      } catch (e) {
        if (!cancelled) { setError(String(e)); setLoading(false); }
      }
      return () => { cancelled = true; };
    }

    // ── Real YAML ──────────────────────────────────────────────
    setIsMock(false);
    Promise.all([loadEnvConfig(inputDir), loadSchedule(inputDir)])
      .then(([env, sched]) => {
        if (!cancelled) setGanttData(buildGanttData(env, sched));
      })
      .catch(e => {
        // Real files missing → degrade gracefully to the mock grid.
        if (!cancelled) {
          try {
            setGanttData(buildGanttData(MOCK_ENV_CONFIG, MOCK_SCHEDULE));
            setIsMock(true);
          } catch {
            setError(String(e));
          }
        }
      })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [inputDir]);

  return { ganttData, loading, error, isMock };
}
