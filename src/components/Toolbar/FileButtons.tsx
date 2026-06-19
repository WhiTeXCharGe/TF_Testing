import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { openTwoYamlFiles, downloadScheduleYaml } from '../../services/fileService';
import { UI } from '../../config/uiText';

const btn: CSSProperties = {
  padding: '4px 10px',
  backgroundColor: '#1976d2',
  color: '#fff',
  border: 'none',
  borderRadius: 3,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
  whiteSpace: 'nowrap',
};

const btnDisabled: CSSProperties = { ...btn, backgroundColor: '#90a4ae', cursor: 'default' };

export function FileButtons() {
  const { state, dispatch } = useAppContext();

  const handleOpen = async () => {
    try {
      const loaded = await openTwoYamlFiles();
      dispatch({
        type: 'LOAD_FILES',
        payload: {
          envConfig: loaded.envConfig,
          schedule: loaded.schedule,
          envPath: loaded.envFileName,
          schedulePath: loaded.scheduleFileName,
        },
      });
    } catch (err: unknown) {
      if (err instanceof Error && !err.message.includes('選択されませんでした')) {
        dispatch({ type: 'SET_ERROR', payload: String(err) });
      }
    }
  };

  const handleSave = () => {
    if (state.schedule) downloadScheduleYaml(state.schedule, state.currentSchedulePath ?? 'Schedule.yaml');
  };

  const hasSchedule = !!state.schedule;

  return (
    <div style={{ display: 'flex', gap: 4 }}>
      <button onClick={handleOpen} style={btn}>{UI.open}</button>
      <button onClick={handleSave} style={hasSchedule ? btn : btnDisabled} disabled={!hasSchedule}>{UI.save}</button>
      <button onClick={handleSave} style={hasSchedule ? { ...btn, backgroundColor: '#388e3c' } : btnDisabled} disabled={!hasSchedule}>{UI.saveAs}</button>
    </div>
  );
}
