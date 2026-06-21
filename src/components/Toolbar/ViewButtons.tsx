import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { ViewMode } from '../../types/appState';

export function ViewButtons() {
  const { state, dispatch } = useAppContext();

  const toggle = (view: ViewMode) => dispatch({ type: 'SWITCH_VIEW', payload: view });

  const base: CSSProperties = {
    padding: '4px 10px',
    border: '1px solid #999',
    borderRadius: 3,
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  };

  return (
    <div style={{ display: 'flex', gap: 0 }}>
      <button
        onClick={() => toggle('worker')}
        style={{ ...base, borderRadius: '3px 0 0 3px', backgroundColor: state.currentView === 'worker' ? '#1976d2' : '#fff', color: state.currentView === 'worker' ? '#fff' : '#333' }}
      >
        {UI.workerView}
      </button>
      <button
        onClick={() => toggle('device')}
        style={{ ...base, borderRadius: '0 3px 3px 0', borderLeft: 'none', backgroundColor: state.currentView === 'device' ? '#1976d2' : '#fff', color: state.currentView === 'device' ? '#fff' : '#333' }}
      >
        {UI.deviceView}
      </button>
    </div>
  );
}
