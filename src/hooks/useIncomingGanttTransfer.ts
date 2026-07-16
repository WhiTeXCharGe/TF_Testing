import { useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { parseEnvConfigYaml, parseScheduleYaml } from '../services/yamlService';
import { SCHEDULER_WEB_URL } from '../config/appConfig';

// Consumes a ?incomingTransfer=<token> query param set by SchedulerWeb's
// 結果を表示 / コピーファイル表示 buttons, fetching the payload from
// SchedulerWeb's own dev server (cross-origin) and loading it into the editor.
export function useIncomingGanttTransfer() {
  const { dispatch } = useAppContext();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('incomingTransfer');
    if (!token) return;

    // Strip the query param immediately so a refresh/back doesn't re-trigger it.
    const url = new URL(window.location.href);
    url.searchParams.delete('incomingTransfer');
    window.history.replaceState({}, '', url.toString());

    (async () => {
      try {
        const res = await fetch(`${SCHEDULER_WEB_URL}/api/handoff/consume/${token}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.ok) {
          throw new Error(data.error ?? `HTTP ${res.status}`);
        }
        const envConfig = parseEnvConfigYaml(data.envYaml);
        const schedule = parseScheduleYaml(data.scheduleYaml);
        dispatch({
          type: 'LOAD_FILES',
          payload: { envConfig, schedule, envPath: 'EnvConfig.yaml', schedulePath: 'Schedule.yaml' },
        });
      } catch (e) {
        dispatch({ type: 'SET_ERROR', payload: `計画管理ツールからのデータ受信に失敗しました: ${String((e as Error).message ?? e)}` });
      }
    })();
  }, [dispatch]);
}
