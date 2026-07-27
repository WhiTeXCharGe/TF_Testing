import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

export function PlanRangeEditDialog() {
  const { state, dispatch } = useAppContext();
  const [isOpen, setIsOpen] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [error, setError] = useState('');

  if (!state.schedule) return null;

  const open = () => {
    setStartDate(state.schedule!.planRange.startDate);
    setEndDate(state.schedule!.planRange.endDate);
    setError('');
    setIsOpen(true);
  };

  const handleApply = () => {
    if (!startDate || !endDate || startDate > endDate) {
      setError(UI.invalidDateRangeMessage);
      return;
    }
    dispatch({ type: 'UPDATE_PLAN_RANGE', payload: { startDate, endDate } });
    setIsOpen(false);
  };

  const triggerBtn: React.CSSProperties = {
    padding: '4px 10px',
    border: '1px solid #ccc',
    borderRadius: 3,
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
    backgroundColor: '#fff',
  };

  const overlay: React.CSSProperties = {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 800,
  };
  const modal: React.CSSProperties = {
    backgroundColor: '#fff', borderRadius: 6, width: 360,
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  };
  const titleBar: React.CSSProperties = {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold',
  };
  const body: React.CSSProperties = { padding: '16px 20px' };
  const fieldRow: React.CSSProperties = { display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 14 };
  const fieldLabel: React.CSSProperties = {
    fontSize: 11, color: '#666', fontWeight: 'bold',
    textTransform: 'uppercase', letterSpacing: 0.5,
  };
  const dateInput: React.CSSProperties = {
    padding: '5px 8px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 12, fontFamily: 'MS Gothic, monospace', width: '100%', boxSizing: 'border-box',
  };
  const footer: React.CSSProperties = {
    display: 'flex', justifyContent: 'flex-end', gap: 8,
    padding: '12px 20px', borderTop: '1px solid #e0e0e0', backgroundColor: '#fafafa',
  };
  const okBtn: React.CSSProperties = {
    padding: '6px 20px', backgroundColor: '#1976d2', color: '#fff',
    border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  };
  const cancelBtn: React.CSSProperties = {
    padding: '6px 16px', border: '1px solid #aaa', borderRadius: 4,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#fff', fontFamily: 'MS Gothic, monospace',
  };

  return (
    <>
      <button style={triggerBtn} onClick={open}>{UI.planRangeEditBtn}</button>

      {isOpen && (
        <div style={overlay} onClick={e => { if (e.target === e.currentTarget) setIsOpen(false); }}>
          <div style={modal}>
            <div style={titleBar}>{UI.planRangeDialogTitle}</div>
            <div style={body}>
              <div style={fieldRow}>
                <span style={fieldLabel}>{UI.planRangeStartLabel}</span>
                <input
                  type="date" style={dateInput}
                  value={startDate}
                  onChange={e => { setStartDate(e.target.value); setError(''); }}
                />
              </div>
              <div style={fieldRow}>
                <span style={fieldLabel}>{UI.planRangeEndLabel}</span>
                <input
                  type="date" style={dateInput}
                  value={endDate}
                  onChange={e => { setEndDate(e.target.value); setError(''); }}
                />
              </div>
              {error && <div style={{ color: '#c62828', fontSize: 11 }}>{error}</div>}
            </div>
            <div style={footer}>
              <button style={okBtn} onClick={handleApply}>{UI.dialogOk}</button>
              <button style={cancelBtn} onClick={() => setIsOpen(false)}>{UI.dialogCancel}</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
