import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { PlanFlexibility } from '../../types/schedule';

export function PlanFlexBulkSettings() {
  const { state, dispatch } = useAppContext();
  const [isOpen, setIsOpen] = useState(false);
  const [target, setTarget] = useState<'all' | 'selected'>('all');
  const [flexibility, setFlexibility] = useState<PlanFlexibility>('Flexible');
  const [useDate, setUseDate] = useState(false);
  const [targetDate, setTargetDate] = useState('');

  if (!state.schedule) return null;

  const hasSelection = state.selectedAssignmentIndex !== null;

  const handleApply = () => {
    dispatch({
      type: 'BULK_UPDATE_FLEXIBILITY',
      payload: {
        flexibility,
        target,
        targetDate: useDate ? targetDate : undefined,
      },
    });
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
    backgroundColor: '#fff', borderRadius: 6, width: 400,
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  };
  const titleBar: React.CSSProperties = {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold',
  };
  const body: React.CSSProperties = { padding: '16px 20px' };
  const section: React.CSSProperties = { marginBottom: 16 };
  const sectionLabel: React.CSSProperties = {
    fontSize: 11, color: '#666', marginBottom: 8, fontWeight: 'bold',
    textTransform: 'uppercase', letterSpacing: 0.5,
  };
  const radioRow: React.CSSProperties = {
    display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0',
    fontSize: 12, cursor: 'pointer',
  };
  const radioDisabled: React.CSSProperties = { ...radioRow, color: '#bbb', cursor: 'default' };
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

  const flexOptions: { value: PlanFlexibility; label: string }[] = [
    { value: 'Flexible', label: UI.flexibleDesc },
    { value: 'Reluctant', label: UI.reluctantDesc },
    { value: 'Fixed', label: UI.fixedDesc },
  ];

  return (
    <>
      <button style={triggerBtn} onClick={() => setIsOpen(true)}>{UI.bulkFlexEdit}</button>

      {isOpen && (
        <div style={overlay} onClick={e => { if (e.target === e.currentTarget) setIsOpen(false); }}>
          <div style={modal}>
            <div style={titleBar}>{UI.bulkDialogTitle}</div>
            <div style={body}>

              {/* 対象 */}
              <div style={section}>
                <div style={sectionLabel}>{UI.bulkTargetLabel}</div>
                <label style={radioRow}>
                  <input
                    type="radio" name="target" value="all"
                    checked={target === 'all'}
                    onChange={() => setTarget('all')}
                  />
                  {UI.bulkTargetAll}
                </label>
                <label style={hasSelection ? radioRow : radioDisabled}>
                  <input
                    type="radio" name="target" value="selected"
                    checked={target === 'selected'}
                    onChange={() => setTarget('selected')}
                    disabled={!hasSelection}
                  />
                  {UI.bulkTargetSelected}
                  {!hasSelection && <span style={{ fontSize: 10, color: '#aaa' }}> {UI.bulkNoSelectionSuffix}</span>}
                </label>
              </div>

              {/* 変更先の柔軟性 */}
              <div style={section}>
                <div style={sectionLabel}>{UI.bulkFlexLabel}</div>
                {flexOptions.map(opt => (
                  <label key={opt.value} style={radioRow}>
                    <input
                      type="radio" name="flex" value={opt.value}
                      checked={flexibility === opt.value}
                      onChange={() => setFlexibility(opt.value)}
                    />
                    {opt.label}
                  </label>
                ))}
              </div>

              {/* 日付フィルター */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={useDate}
                    onChange={e => setUseDate(e.target.checked)}
                  />
                  {UI.bulkDateFilter}
                </label>
                {useDate && (
                  <input
                    type="date"
                    value={targetDate}
                    onChange={e => setTargetDate(e.target.value)}
                    style={{
                      padding: '4px 8px', border: '1px solid #ccc', borderRadius: 3,
                      fontSize: 12, marginLeft: 20, width: 160,
                    }}
                  />
                )}
              </div>

            </div>
            <div style={footer}>
              <button style={okBtn} onClick={handleApply}>{UI.bulkApply}</button>
              <button style={cancelBtn} onClick={() => setIsOpen(false)}>{UI.dialogCancel}</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
