import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { PlanFlexibility, Assignment, WorkDate } from '../../types/schedule';
import { HOURS_PER_DAY } from '../../config/appConfig';
import { generateDateRange, isWeekend } from '../../utils/dateUtils';
import { Workflow } from '../../types/envConfig';

// ── Per-assignment entry ──────────────────────────────────────────────────────

interface Entry {
  key: string;
  deviceId: string;
  phaseId: string;
  opTaskId: string;
  workerId: string;
  startDate: string;
  endDate: string;
  hoursPerDay: number;
  flexibility: PlanFlexibility;
}

function makeEntry(startDate: string): Entry {
  return {
    key: `ae_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
    deviceId: '', phaseId: '', opTaskId: '', workerId: '',
    startDate, endDate: '',
    hoursPerDay: HOURS_PER_DAY,
    flexibility: 'Flexible',
  };
}

// ── Dialog ────────────────────────────────────────────────────────────────────

export function TaskAddDialog() {
  const { state, dispatch } = useAppContext();

  // All hooks must be before any early return
  const planStart = state.schedule?.planRange.startDate ?? '';
  const [entries, setEntries] = useState<Entry[]>([makeEntry(planStart)]);

  const { isTaskAddDialogOpen, schedule, envConfig } = state;

  if (!isTaskAddDialogOpen || !schedule || !envConfig) return null;

  // ── Entry helpers ────────────────────────────────────────────────────────

  const addEntry = () =>
    setEntries(prev => [...prev, makeEntry(planStart)]);

  const removeEntry = (key: string) =>
    setEntries(prev => prev.length > 1 ? prev.filter(e => e.key !== key) : prev);

  const patch = (key: string, p: Partial<Entry>) =>
    setEntries(prev => prev.map(e => e.key === key ? { ...e, ...p } : e));

  const handleDeviceChange = (key: string, deviceId: string) =>
    patch(key, { deviceId, phaseId: '', opTaskId: '' });

  const handlePhaseChange = (key: string, phaseId: string) =>
    patch(key, { phaseId, opTaskId: '' });

  const handleOpChange = (key: string, opTaskId: string, workflowList: Workflow[]) => {
    const entry = entries.find(e => e.key === key);
    if (!entry) return;
    const device = schedule.workflowTaskList.find(d => d.id === entry.deviceId);
    const phase = device?.phaseTaskList.find(p => p.id === entry.phaseId);
    const op = phase?.operationTaskList.find(o => o.id === opTaskId);
    const envOp = op ? findEnvOp(op.operation, workflowList) : undefined;
    const hoursPerDay = envOp?.workHours?.[0] ?? HOURS_PER_DAY;
    patch(key, { opTaskId, hoursPerDay });
  };

  // ── Submit ────────────────────────────────────────────────────────────────

  const handleOk = () => {
    const valid = entries.filter(e =>
      e.deviceId && e.phaseId && e.opTaskId && e.workerId && e.startDate && e.endDate
    );
    if (valid.length === 0) return;

    for (const e of valid) {
      const workDateList: WorkDate[] = generateDateRange(e.startDate, e.endDate)
        .filter(d => !isWeekend(d))
        .map(d => ({ date: d, hour: e.hoursPerDay }));

      const assignment: Assignment = {
        worker: e.workerId,
        operationTask: e.opTaskId,
        startDate: e.startDate,
        endDate: e.endDate,
        workDateList,
        planFlexibility: e.flexibility,
      };
      dispatch({ type: 'ADD_ASSIGNMENT', payload: assignment });
    }

    dispatch({ type: 'CLOSE_TASK_ADD_DIALOG' });
    setEntries([makeEntry(planStart)]);
  };

  const handleClose = () => {
    dispatch({ type: 'CLOSE_TASK_ADD_DIALOG' });
    setEntries([makeEntry(planStart)]);
  };

  const canSubmit = entries.some(e =>
    e.deviceId && e.phaseId && e.opTaskId && e.workerId && e.startDate && e.endDate
  );

  // ── Render ────────────────────────────────────────────────────────────────

  const S = styles;

  return (
    <div style={S.overlay} onClick={e => e.target === e.currentTarget && handleClose()}>
      <div style={S.modal}>
        <div style={S.titleBar}>割付追加</div>

        <div style={S.body}>
          {entries.map((entry, idx) => {
            // Compute cascading options for this entry
            const device = schedule.workflowTaskList.find(d => d.id === entry.deviceId);
            const phases = device?.phaseTaskList ?? [];
            const phase = phases.find(p => p.id === entry.phaseId);
            const opTasks = phase?.operationTaskList ?? [];
            const selectedOp = opTasks.find(o => o.id === entry.opTaskId);
            const envOp = selectedOp ? findEnvOp(selectedOp.operation, envConfig.workflowList) : undefined;
            const hoursOptions = envOp?.workHours?.length ? envOp.workHours : [4, 6, 8, 10, 12];

            return (
              <div key={entry.key} style={S.card}>
                {/* Card header */}
                <div style={S.cardHeader}>
                  <span style={S.cardTitle}>割付 {idx + 1}</span>
                  <button
                    style={{ ...S.iconBtn, color: '#b71c1c' }}
                    onClick={() => removeEntry(entry.key)}
                    disabled={entries.length === 1}
                    title="削除"
                  >
                    −
                  </button>
                </div>

                <div style={S.cardBody}>
                  {/* Row: 装置 / 工程 / 作業 */}
                  <div style={S.row3}>
                    <Field label="装置 *">
                      <select style={S.input} value={entry.deviceId}
                        onChange={e => handleDeviceChange(entry.key, e.target.value)}>
                        <option value="">---</option>
                        {schedule.workflowTaskList.map(d => (
                          <option key={d.id} value={d.id}>{d.name ?? d.id}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="工程 *">
                      <select style={S.input} value={entry.phaseId}
                        disabled={!entry.deviceId}
                        onChange={e => handlePhaseChange(entry.key, e.target.value)}>
                        <option value="">---</option>
                        {phases.map(p => (
                          <option key={p.id} value={p.id}>{p.name ?? p.id}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="作業 *">
                      <select style={S.input} value={entry.opTaskId}
                        disabled={!entry.phaseId}
                        onChange={e => handleOpChange(entry.key, e.target.value, envConfig.workflowList)}>
                        <option value="">---</option>
                        {opTasks.map(o => (
                          <option key={o.id} value={o.id}>{o.name ?? o.operation}</option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  {/* Row: 作業者 / 作業時間 / 計画柔軟性 */}
                  <div style={S.row3}>
                    <Field label="作業者 *">
                      <select style={S.input} value={entry.workerId}
                        onChange={e => patch(entry.key, { workerId: e.target.value })}>
                        <option value="">---</option>
                        {envConfig.workerList.map(w => (
                          <option key={w.id} value={w.id}>{w.name ?? w.id}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="作業時間 (時間/日)">
                      <select style={S.input} value={entry.hoursPerDay}
                        onChange={e => patch(entry.key, { hoursPerDay: Number(e.target.value) })}>
                        {hoursOptions.map(h => (
                          <option key={h} value={h}>{h} 時間/日</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="計画柔軟性">
                      <select style={S.input} value={entry.flexibility}
                        onChange={e => patch(entry.key, { flexibility: e.target.value as PlanFlexibility })}>
                        <option value="Flexible">Flexible</option>
                        <option value="Reluctant">Reluctant</option>
                        <option value="Fixed">Fixed</option>
                      </select>
                    </Field>
                  </div>

                  {/* Row: 開始日 / 終了日 */}
                  <div style={{ display: 'flex', gap: 10 }}>
                    <Field label="開始日 *">
                      <input style={S.input} type="date" value={entry.startDate}
                        onChange={e => patch(entry.key, { startDate: e.target.value })} />
                    </Field>
                    <Field label="終了日 *">
                      <input style={S.input} type="date" value={entry.endDate}
                        onChange={e => patch(entry.key, { endDate: e.target.value })} />
                    </Field>
                    <div style={{ flex: 1 }} />
                  </div>
                </div>
              </div>
            );
          })}

          <button style={S.addBtn} onClick={addEntry}>+ 割付追加</button>
        </div>

        <div style={S.footer}>
          <button style={canSubmit ? S.primaryBtn : S.disabledBtn}
            onClick={handleOk} disabled={!canSubmit}>
            OK
          </button>
          <button style={S.cancelBtn} onClick={handleClose}>キャンセル</button>
        </div>
      </div>
    </div>
  );
}

// ── Helper ────────────────────────────────────────────────────────────────────

function findEnvOp(operationId: string, workflowList: Workflow[]) {
  for (const wf of workflowList) {
    for (const ph of wf.phaseList) {
      const op = ph.operationList.find(o => o.id === operationId);
      if (op) return op;
    }
  }
  return undefined;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>
        {label}
      </label>
      {children}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  overlay: {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 900,
  } as React.CSSProperties,
  modal: {
    backgroundColor: '#fff', borderRadius: 6, width: 640,
    maxHeight: '88vh', display: 'flex', flexDirection: 'column',
    boxShadow: '0 8px 28px rgba(0,0,0,0.35)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  } as React.CSSProperties,
  titleBar: {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold', flexShrink: 0,
  } as React.CSSProperties,
  body: {
    padding: 14, flex: 1, overflowY: 'auto',
  } as React.CSSProperties,
  footer: {
    display: 'flex', justifyContent: 'flex-end', gap: 8,
    padding: '10px 14px', borderTop: '1px solid #e0e0e0',
    backgroundColor: '#fafafa', flexShrink: 0,
  } as React.CSSProperties,
  card: {
    border: '1px solid #d0d5dd', borderRadius: 4,
    marginBottom: 10, backgroundColor: '#fafbfc', overflow: 'hidden',
  } as React.CSSProperties,
  cardHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '5px 10px', backgroundColor: '#e8eef6', borderBottom: '1px solid #d0d5dd',
  } as React.CSSProperties,
  cardTitle: {
    fontSize: 12, fontWeight: 'bold', color: '#1c2b3a',
  } as React.CSSProperties,
  cardBody: { padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 } as React.CSSProperties,
  iconBtn: {
    padding: '1px 8px', border: '1px solid #ccc', borderRadius: 3,
    cursor: 'pointer', fontSize: 13, backgroundColor: '#fff',
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  row3: {
    display: 'flex', gap: 8,
  } as React.CSSProperties,
  input: {
    width: '100%', padding: '4px 5px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 11, boxSizing: 'border-box', fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  addBtn: {
    width: '100%', padding: 8, border: '2px dashed #ccc', borderRadius: 4,
    backgroundColor: 'transparent', cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace', color: '#666',
  } as React.CSSProperties,
  primaryBtn: {
    padding: '6px 20px', backgroundColor: '#1976d2', color: '#fff',
    border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  disabledBtn: {
    padding: '6px 20px', backgroundColor: '#aaa', color: '#fff',
    border: 'none', borderRadius: 4, cursor: 'default', fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  cancelBtn: {
    padding: '6px 16px', border: '1px solid #aaa', borderRadius: 4,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#fff',
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
};
