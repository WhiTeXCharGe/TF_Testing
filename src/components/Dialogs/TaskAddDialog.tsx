import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { PlanFlexibility, Assignment, WorkDate } from '../../types/schedule';
import { HOURS_PER_DAY } from '../../config/appConfig';
import { generateDateRange, isWeekend } from '../../utils/dateUtils';
import { Workflow } from '../../types/envConfig';
import { SearchableSelect, SelectOption } from '../common/SearchableSelect';

const MISC_WORKFLOW_ID = 'wf_misc';

// ── Types ─────────────────────────────────────────────────────────────────────

interface WorkerDateEntry {
  workerId: string;
  startDate: string;
  endDate: string;
  hoursPerDay: number;
  flexibility: PlanFlexibility;
}

interface RegularEntry {
  deviceId: string;
  phaseId: string;
  opTaskId: string;
  workerDates: WorkerDateEntry[];
}

interface MiscEntry {
  miscTaskId: string;
  isNewTask: boolean;
  newTaskName: string;
  workerDates: WorkerDateEntry[];
}

function makeWorkerDate(planStart: string): WorkerDateEntry {
  return { workerId: '', startDate: planStart, endDate: '', hoursPerDay: HOURS_PER_DAY, flexibility: 'Flexible' };
}

// ── Dialog ────────────────────────────────────────────────────────────────────

export function TaskAddDialog() {
  const { state, dispatch } = useAppContext();
  const { isTaskAddDialogOpen, schedule, envConfig } = state;
  const planStart = schedule?.planRange.startDate ?? '';

  const [addType, setAddType] = useState<'regular' | 'misc' | 'unavailable'>('regular');

  const [regular, setRegular] = useState<RegularEntry>({
    deviceId: '', phaseId: '', opTaskId: '',
    workerDates: [makeWorkerDate(planStart)],
  });

  const [misc, setMisc] = useState<MiscEntry>({
    miscTaskId: '',
    isNewTask: false,
    newTaskName: '',
    workerDates: [makeWorkerDate(planStart)],
  });

  const [unavailWorkerIds, setUnavailWorkerIds] = useState<string[]>([]);
  const [unavailStart, setUnavailStart] = useState(planStart);
  const [unavailEnd, setUnavailEnd] = useState('');
  const [unavailWorkerSearch, setUnavailWorkerSearch] = useState('');

  if (!isTaskAddDialogOpen || !schedule || !envConfig) return null;

  const S = styles;

  const miscWorkflowTasks = schedule.workflowTaskList.filter(wt => wt.workflow === MISC_WORKFLOW_ID);

  // ── Regular helpers ────────────────────────────────────────────────────────
  const regDevice = schedule.workflowTaskList.find(d => d.id === regular.deviceId);
  const regPhases = regDevice?.phaseTaskList ?? [];
  const regPhase = regPhases.find(p => p.id === regular.phaseId);
  const regOpTasks = regPhase?.operationTaskList ?? [];
  const regSelectedOp = regOpTasks.find(o => o.id === regular.opTaskId);
  const envOp = regSelectedOp ? findEnvOp(regSelectedOp.operation, envConfig.workflowList) : undefined;
  const hoursOptions = envOp?.workHours?.length ? envOp.workHours : [4, 6, 8, 10, 12];

  const regularDevices = schedule.workflowTaskList.filter(wt => wt.workflow !== MISC_WORKFLOW_ID);

  const patchWorkerDate = (
    list: WorkerDateEntry[],
    idx: number,
    p: Partial<WorkerDateEntry>,
  ): WorkerDateEntry[] => list.map((wd, i) => i === idx ? { ...wd, ...p } : wd);

  // ── Submit ────────────────────────────────────────────────────────────────

  const handleOk = () => {
    if (addType === 'unavailable') {
      if (unavailWorkerIds.length === 0 || !unavailStart || !unavailEnd || unavailStart > unavailEnd) return;
      const dates = generateDateRange(unavailStart, unavailEnd);
      dispatch({
        type: 'ADD_UNAVAILABLE_DATES',
        payload: unavailWorkerIds.map(wid => ({ workerId: wid, dates })),
      });
      dispatch({ type: 'CLOSE_TASK_ADD_DIALOG' });
      resetState();
      return;
    }
    if (addType === 'regular') {
      if (!regular.deviceId || !regular.phaseId || !regular.opTaskId) return;
      const valid = regular.workerDates.filter(wd => wd.workerId && wd.startDate && wd.endDate);
      if (valid.length === 0) return;
      for (const wd of valid) {
        const workDateList: WorkDate[] = generateDateRange(wd.startDate, wd.endDate)
          .filter(d => !isWeekend(d))
          .map(d => ({ date: d, hour: wd.hoursPerDay }));
        dispatch({
          type: 'ADD_ASSIGNMENT',
          payload: {
            worker: wd.workerId,
            operationTask: regular.opTaskId,
            startDate: wd.startDate,
            endDate: wd.endDate,
            workDateList,
            planFlexibility: wd.flexibility,
          } as Assignment,
        });
      }
    } else if (addType === 'misc') {
      const opTaskId = misc.isNewTask ? '' : misc.miscTaskId;
      let resolvedOpTaskId = opTaskId;

      if (misc.isNewTask && misc.newTaskName) {
        const newId = `misc_${Date.now().toString(36)}`;
        dispatch({
          type: 'ADD_WORKFLOW_TASKS',
          payload: [{
            id: newId,
            name: misc.newTaskName,
            workflow: MISC_WORKFLOW_ID,
            phaseTaskList: [],
          }],
        });
        resolvedOpTaskId = newId;
      }

      if (!resolvedOpTaskId) return;
      const valid = misc.workerDates.filter(wd => wd.workerId && wd.startDate && wd.endDate);
      if (valid.length === 0) return;
      for (const wd of valid) {
        const workDateList: WorkDate[] = generateDateRange(wd.startDate, wd.endDate)
          .filter(d => !isWeekend(d))
          .map(d => ({ date: d, hour: wd.hoursPerDay }));
        dispatch({
          type: 'ADD_ASSIGNMENT',
          payload: {
            worker: wd.workerId,
            operationTask: resolvedOpTaskId,
            startDate: wd.startDate,
            endDate: wd.endDate,
            workDateList,
            planFlexibility: wd.flexibility,
          } as Assignment,
        });
      }
    }

    dispatch({ type: 'CLOSE_TASK_ADD_DIALOG' });
    resetState();
  };

  const resetState = () => {
    setRegular({ deviceId: '', phaseId: '', opTaskId: '', workerDates: [makeWorkerDate(planStart)] });
    setMisc({ miscTaskId: '', isNewTask: false, newTaskName: '', workerDates: [makeWorkerDate(planStart)] });
    setUnavailWorkerIds([]);
    setUnavailStart(planStart);
    setUnavailEnd('');
    setUnavailWorkerSearch('');
    setAddType('regular');
  };

  const handleClose = () => {
    dispatch({ type: 'CLOSE_TASK_ADD_DIALOG' });
    resetState();
  };

  const canSubmit = addType === 'regular'
    ? !!(regular.deviceId && regular.phaseId && regular.opTaskId && regular.workerDates.some(w => w.workerId && w.startDate && w.endDate))
    : addType === 'misc'
    ? !!(
        (misc.isNewTask ? misc.newTaskName : misc.miscTaskId) &&
        misc.workerDates.some(w => w.workerId && w.startDate && w.endDate)
      )
    : !!(unavailWorkerIds.length > 0 && unavailStart && unavailEnd && unavailStart <= unavailEnd);

  return (
    <div style={S.overlay} onClick={e => e.target === e.currentTarget && handleClose()}>
      <div style={S.modal}>
        <div style={S.titleBar}>割付追加</div>

        <div style={S.body}>
          {/* Step 1: Add type */}
          <div style={S.card}>
            <div style={S.cardHeader}><span style={S.cardTitle}>追加タイプ選択</span></div>
            <div style={{ ...S.cardBody, flexDirection: 'row', gap: 16 }}>
              <label style={S.radioLabel}>
                <input type="radio" checked={addType === 'regular'} onChange={() => setAddType('regular')} />
                <span style={{ marginLeft: 6 }}>通常ワークフロー</span>
              </label>
              <label style={S.radioLabel}>
                <input type="radio" checked={addType === 'misc'} onChange={() => setAddType('misc')} />
                <span style={{ marginLeft: 6 }}>その他作業 (Misc)</span>
              </label>
              <label style={S.radioLabel}>
                <input type="radio" checked={addType === 'unavailable'} onChange={() => setAddType('unavailable')} />
                <span style={{ marginLeft: 6 }}>休日・不在設定</span>
              </label>
            </div>
          </div>

          {/* Regular flow */}
          {addType === 'regular' && (
            <div style={S.card}>
              <div style={S.cardHeader}><span style={S.cardTitle}>装置・工程・作業 選択</span></div>
              <div style={S.cardBody}>
                <div style={S.row3}>
                  <Field label="装置 *">
                    <SearchableSelect
                      value={regular.deviceId}
                      options={regularDevices.map(d => ({ value: d.id, label: d.name ?? d.id }))}
                      onChange={v => setRegular(r => ({ ...r, deviceId: v, phaseId: '', opTaskId: '' }))}
                    />
                  </Field>
                  <Field label="工程 *">
                    <SearchableSelect
                      value={regular.phaseId}
                      options={regPhases.map(p => ({ value: p.id, label: p.name ?? p.id }))}
                      onChange={v => setRegular(r => ({ ...r, phaseId: v, opTaskId: '' }))}
                      disabled={!regular.deviceId}
                    />
                  </Field>
                  <Field label="作業 *">
                    <SearchableSelect
                      value={regular.opTaskId}
                      options={regOpTasks.map(o => ({ value: o.id, label: o.name ?? o.operation }))}
                      onChange={v => {
                        const op = regOpTasks.find(o => o.id === v);
                        const envO = op ? findEnvOp(op.operation, envConfig.workflowList) : undefined;
                        const hpd = envO?.workHours?.[0] ?? HOURS_PER_DAY;
                        setRegular(r => ({
                          ...r,
                          opTaskId: v,
                          workerDates: r.workerDates.map(wd => ({ ...wd, hoursPerDay: hpd })),
                        }));
                      }}
                      disabled={!regular.phaseId}
                    />
                  </Field>
                </div>

                <div style={{ borderTop: '1px solid #e8e8e8', paddingTop: 8, marginTop: 4 }}>
                  <div style={{ fontSize: 11, color: '#555', fontWeight: 'bold', marginBottom: 6 }}>作業者・日程</div>
                  {regular.workerDates.map((wd, idx) => (
                    <WorkerDateRow
                      key={idx}
                      idx={idx}
                      wd={wd}
                      hoursOptions={hoursOptions}
                      canRemove={regular.workerDates.length > 1}
                      onChange={p => setRegular(r => ({ ...r, workerDates: patchWorkerDate(r.workerDates, idx, p) }))}
                      onRemove={() => setRegular(r => ({ ...r, workerDates: r.workerDates.filter((_, i) => i !== idx) }))}
                    />
                  ))}
                  <button style={S.addWorkerBtn}
                    onClick={() => setRegular(r => ({ ...r, workerDates: [...r.workerDates, makeWorkerDate(planStart)] }))}>
                    + 作業者追加
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Misc flow */}
          {addType === 'misc' && (
            <div style={S.card}>
              <div style={S.cardHeader}><span style={S.cardTitle}>その他作業 選択</span></div>
              <div style={S.cardBody}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                  <label style={S.radioLabel}>
                    <input type="radio" checked={!misc.isNewTask} onChange={() => setMisc(m => ({ ...m, isNewTask: false }))} />
                    <span style={{ marginLeft: 6 }}>既存のタスク</span>
                  </label>
                  <label style={S.radioLabel}>
                    <input type="radio" checked={misc.isNewTask} onChange={() => setMisc(m => ({ ...m, isNewTask: true }))} />
                    <span style={{ marginLeft: 6 }}>新規タスク作成</span>
                  </label>
                </div>

                {!misc.isNewTask ? (
                  <Field label="タスク選択 *">
                    <SearchableSelect
                      value={misc.miscTaskId}
                      options={miscWorkflowTasks.map(wt => ({ value: wt.id, label: wt.name ?? wt.id }))}
                      onChange={v => setMisc(m => ({ ...m, miscTaskId: v }))}
                    />
                  </Field>
                ) : (
                  <Field label="新タスク名 *">
                    <input style={S.input} type="text" value={misc.newTaskName}
                      placeholder="タスク名を入力"
                      onChange={e => setMisc(m => ({ ...m, newTaskName: e.target.value }))} />
                  </Field>
                )}

                <div style={{ borderTop: '1px solid #e8e8e8', paddingTop: 8, marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: '#555', fontWeight: 'bold', marginBottom: 6 }}>作業者・日程</div>
                  {misc.workerDates.map((wd, idx) => (
                    <WorkerDateRow
                      key={idx}
                      idx={idx}
                      wd={wd}
                      hoursOptions={[4, 6, 8, 10, 12]}
                      canRemove={misc.workerDates.length > 1}
                      onChange={p => setMisc(m => ({ ...m, workerDates: patchWorkerDate(m.workerDates, idx, p) }))}
                      onRemove={() => setMisc(m => ({ ...m, workerDates: m.workerDates.filter((_, i) => i !== idx) }))}
                    />
                  ))}
                  <button style={S.addWorkerBtn}
                    onClick={() => setMisc(m => ({ ...m, workerDates: [...m.workerDates, makeWorkerDate(planStart)] }))}>
                    + 作業者追加
                  </button>
                </div>
              </div>
            </div>
          )}
          {/* Unavailable flow */}
          {addType === 'unavailable' && (
            <div style={S.card}>
              <div style={S.cardHeader}><span style={S.cardTitle}>休日・不在設定</span></div>
              <div style={S.cardBody}>
                <div style={{ display: 'flex', gap: 12 }}>
                  <Field label="開始日 *">
                    <input style={S.input} type="date" value={unavailStart}
                      onChange={e => setUnavailStart(e.target.value)} />
                  </Field>
                  <Field label="終了日 *">
                    <input style={S.input} type="date" value={unavailEnd}
                      onChange={e => setUnavailEnd(e.target.value)} />
                  </Field>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#555', fontWeight: 'bold', marginBottom: 6 }}>対象作業者（複数選択可） *</div>
                  <input
                    type="text"
                    placeholder="作業者を検索..."
                    value={unavailWorkerSearch}
                    onChange={e => setUnavailWorkerSearch(e.target.value)}
                    style={{ ...S.input, marginBottom: 4 }}
                  />
                  <div style={{
                    maxHeight: 180, overflowY: 'auto', border: '1px solid #ccc', borderRadius: 3,
                    padding: '4px 8px', background: '#fff',
                  }}>
                    {(envConfig?.workerList ?? [])
                      .filter(w => {
                        if (!unavailWorkerSearch.trim()) return true;
                        const q = unavailWorkerSearch.toLowerCase();
                        const company = envConfig?.workerCompanyList.find(c => c.id === w.workerCompany);
                        return (w.name ?? w.id).toLowerCase().includes(q) || (company?.name ?? '').toLowerCase().includes(q);
                      })
                      .map(w => {
                      const company = envConfig?.workerCompanyList.find(c => c.id === w.workerCompany);
                      const checked = unavailWorkerIds.includes(w.id);
                      return (
                        <label key={w.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 0', cursor: 'pointer', fontSize: 11 }}>
                          <input type="checkbox" checked={checked}
                            onChange={() => setUnavailWorkerIds(prev =>
                              checked ? prev.filter(id => id !== w.id) : [...prev, w.id]
                            )}
                          />
                          <span style={{ fontWeight: 'bold', color: '#1c2b3a' }}>{w.name ?? w.id}</span>
                          {company && <span style={{ color: '#888' }}>({company.name ?? w.workerCompany})</span>}
                        </label>
                      );
                    })}
                  </div>
                  <div style={{ fontSize: 10, color: '#888', marginTop: 4 }}>
                    {unavailWorkerIds.length > 0 ? `${unavailWorkerIds.length}人選択中` : '作業者を選択してください'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div style={S.footer}>
          <button style={canSubmit ? S.primaryBtn : S.disabledBtn} onClick={handleOk} disabled={!canSubmit}>OK</button>
          <button style={S.cancelBtn} onClick={handleClose}>キャンセル</button>
        </div>
      </div>
    </div>
  );
}

// ── Worker Date Row ────────────────────────────────────────────────────────────

function WorkerDateRow({
  idx, wd, hoursOptions, canRemove, onChange, onRemove,
}: {
  idx: number;
  wd: WorkerDateEntry;
  hoursOptions: number[];
  canRemove: boolean;
  onChange: (p: Partial<WorkerDateEntry>) => void;
  onRemove: () => void;
}) {
  const { state } = useAppContext();
  const { envConfig } = state;
  const S = styles;

  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end', marginBottom: 6 }}>
      <div style={{ flex: 2, minWidth: 0 }}>
        <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>
          作業者 {idx + 1} *
        </label>
        <SearchableSelect
          value={wd.workerId}
          options={(envConfig?.workerList ?? []).map(w => {
            const company = envConfig?.workerCompanyList.find(c => c.id === w.workerCompany);
            return { value: w.id, label: w.name ?? w.id, sub: company?.name ?? w.workerCompany ?? '' } as SelectOption;
          })}
          onChange={v => onChange({ workerId: v })}
        />
      </div>
      <div style={{ flex: 1.5 }}>
        <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>開始日 *</label>
        <input style={S.input} type="date" value={wd.startDate} onChange={e => onChange({ startDate: e.target.value })} />
      </div>
      <div style={{ flex: 1.5 }}>
        <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>終了日 *</label>
        <input style={S.input} type="date" value={wd.endDate} onChange={e => onChange({ endDate: e.target.value })} />
      </div>
      <div style={{ flex: 1 }}>
        <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>時間/日</label>
        <select style={S.input} value={wd.hoursPerDay} onChange={e => onChange({ hoursPerDay: Number(e.target.value) })}>
          {hoursOptions.map(h => <option key={h} value={h}>{h}h</option>)}
        </select>
      </div>
      <div style={{ flex: 1 }}>
        <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>柔軟性</label>
        <select style={S.input} value={wd.flexibility} onChange={e => onChange({ flexibility: e.target.value as PlanFlexibility })}>
          <option value="Flexible">Flex</option>
          <option value="Reluctant">Rel</option>
          <option value="Fixed">Fix</option>
        </select>
      </div>
      {canRemove && (
        <button onClick={onRemove} style={{ ...styles.iconBtn, color: '#b71c1c', alignSelf: 'flex-end', marginBottom: 1 }} title="削除">
          −
        </button>
      )}
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
      <label style={{ display: 'block', fontSize: 10, color: '#666', fontWeight: 'bold', marginBottom: 2 }}>{label}</label>
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
    backgroundColor: '#fff', borderRadius: 6, width: 720,
    maxHeight: '88vh', display: 'flex', flexDirection: 'column',
    boxShadow: '0 8px 28px rgba(0,0,0,0.35)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  } as React.CSSProperties,
  titleBar: {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold', flexShrink: 0,
  } as React.CSSProperties,
  body: { padding: 14, flex: 1, overflowY: 'auto' } as React.CSSProperties,
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
  cardTitle: { fontSize: 12, fontWeight: 'bold', color: '#1c2b3a' } as React.CSSProperties,
  cardBody: { padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 } as React.CSSProperties,
  iconBtn: {
    padding: '1px 8px', border: '1px solid #ccc', borderRadius: 3,
    cursor: 'pointer', fontSize: 13, backgroundColor: '#fff',
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  row3: { display: 'flex', gap: 8 } as React.CSSProperties,
  input: {
    width: '100%', padding: '4px 5px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 11, boxSizing: 'border-box', fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  addWorkerBtn: {
    width: '100%', padding: 6, border: '1px dashed #aaa', borderRadius: 4,
    backgroundColor: 'transparent', cursor: 'pointer', fontSize: 11,
    fontFamily: 'MS Gothic, monospace', color: '#555',
  } as React.CSSProperties,
  radioLabel: {
    display: 'flex', alignItems: 'center', fontSize: 12, cursor: 'pointer',
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
