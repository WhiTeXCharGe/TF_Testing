import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

export function DisplayPeriodInput() {
  const { state, dispatch } = useAppContext();
  const start = state.displayStartDate ?? state.schedule?.planRange.startDate ?? '';
  const end = state.displayEndDate ?? state.schedule?.planRange.endDate ?? '';

  const label: CSSProperties = { fontSize: 12, fontFamily: 'MS Gothic, monospace' };
  const input: CSSProperties = {
    padding: '3px 6px',
    border: '1px solid #999',
    borderRadius: 3,
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  };

  const handleChange = (field: 'start' | 'end', value: string) => {
    const newStart = field === 'start' ? value : start;
    const newEnd = field === 'end' ? value : end;
    if (newStart && newEnd) dispatch({ type: 'SET_DISPLAY_PERIOD', payload: { startDate: newStart, endDate: newEnd } });
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={label}>{UI.startDateLabel}</span>
      <input style={input} type="date" value={start} onChange={e => handleChange('start', e.target.value)} />
      <span style={label}>{UI.periodSeparator}</span>
      <span style={label}>{UI.endDateLabel}</span>
      <input style={input} type="date" value={end} onChange={e => handleChange('end', e.target.value)} />
    </div>
  );
}
