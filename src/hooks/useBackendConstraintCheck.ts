import { useAppContext } from '../context/AppContext';

export function useBackendConstraintCheck() {
  const { state, dispatch } = useAppContext();

  const runCheck = async () => {
    if (!state.schedule || !state.envConfig) return;

    dispatch({ type: 'SET_CONSTRAINT_CHECKING', payload: true });
    dispatch({ type: 'OPEN_CONSTRAINT_DIALOG' });

    try {
      const res = await fetch('/api/check-constraints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          envConfig: state.envConfig,
          schedule: state.schedule,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        dispatch({ type: 'SET_ERROR', payload: `制約チェックエラー: ${err.error ?? res.statusText}` });
        dispatch({ type: 'SET_CONSTRAINT_CHECKING', payload: false });
        return;
      }

      const data = await res.json();
      dispatch({
        type: 'SET_BACKEND_VIOLATIONS',
        payload: { violations: data.violations, checkedAt: data.checkedAt },
      });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: 'バックエンドに接続できません。サーバーが起動しているか確認してください。' });
      dispatch({ type: 'SET_CONSTRAINT_CHECKING', payload: false });
    }
  };

  return { runCheck, isChecking: state.isConstraintChecking };
}
