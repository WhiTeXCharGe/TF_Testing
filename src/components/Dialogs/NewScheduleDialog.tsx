import { useState, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { parseEnvConfigYaml, parseScheduleYaml } from '../../services/yamlService';
import { WorkflowTask, PhaseTask, OperationTask } from '../../types/schedule';
import { Workflow } from '../../types/envConfig';
import { SearchableSelect, SelectOption } from '../common/SearchableSelect';
import { UI } from '../../config/uiText';

// ── Per-製番 form entry ───────────────────────────────────────────────────────

interface OpEntry {
  workloadHours: number;
  minWorker: number;
  maxWorker: number;
}

interface WTEntry {
  key: string;
  name: string;
  workflowId: string;
  fabId: string;
  startDate: string;
  phaseEndDates: string[];  // one per phase
  collapsed: boolean;
  opEntries: OpEntry[][];   // [phaseIdx][opIdx]
}

function makeEntry(startDate: string): WTEntry {
  return {
    key: `e_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
    name: '', workflowId: '', fabId: '',
    startDate, phaseEndDates: [],
    collapsed: false, opEntries: [],
  };
}

function initOpEntries(workflow: Workflow): OpEntry[][] {
  return workflow.phaseList.map(phase =>
    phase.operationList.map(op => ({
      workloadHours: op.workloadHours ?? op.workHours?.[0] ?? 8,
      minWorker: op.minWorkerNum ?? 1,
      maxWorker: op.maxWorkerNum ?? 1,
    }))
  );
}

function buildWorkflowTask(entry: WTEntry, workflow: Workflow): WorkflowTask {
  const base = entry.name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase().slice(0, 12)
    + '_' + Date.now().toString(36).slice(-4);
  return {
    id: base,
    name: entry.name,
    workflow: entry.workflowId,
    fab: entry.fabId || undefined,
    phaseTaskList: workflow.phaseList.map((phase, pi): PhaseTask => {
      const ptId = `${base}_p${pi}`;
      return {
        id: ptId,
        name: phase.name,
        phase: phase.id,
        startDate: entry.startDate,
        endDate: entry.phaseEndDates[pi] ?? entry.startDate,
        operationTaskList: phase.operationList.map((op, oi): OperationTask => {
          const oe = entry.opEntries[pi]?.[oi];
          return {
            id: `${ptId}_o${oi}`,
            name: op.name,
            operation: op.id,
            workloadHours: oe?.workloadHours ?? op.workloadHours ?? op.workHours?.[0] ?? 8,
            recommendsWorkerMin: oe?.minWorker ?? op.minWorkerNum ?? 1,
            recommendsWorkerMax: oe?.maxWorker ?? op.maxWorkerNum ?? 1,
          };
        }),
      };
    }),
  };
}

// ── Main component ────────────────────────────────────────────────────────────

export function NewScheduleDialog() {
  const { state, dispatch } = useAppContext();
  const { envConfig, schedule } = state;

  const [tab, setTab] = useState<'upload' | 'form'>('form');

  // Upload tab
  const [schedFile, setSchedFile] = useState<File | null>(null);
  const [envFile, setEnvFile] = useState<File | null>(null);
  const schedRef = useRef<HTMLInputElement>(null);
  const envRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  // Form tab
  const planStart = schedule?.planRange.startDate ?? '';
  const [entries, setEntries] = useState<WTEntry[]>([makeEntry(planStart)]);

  if (!state.isNewScheduleDialogOpen) return null;

  // Workflows that have actual phases (filter out wf_misc / empty phaseList)
  const validWorkflows = envConfig?.workflowList.filter(w => w.phaseList && w.phaseList.length > 0) ?? [];

  const workflowOptions: SelectOption[] = validWorkflows.map(w => ({ value: w.id, label: w.name ?? w.id }));
  const fabOptions: SelectOption[] = [
    { value: '', label: UI.noneOptionLabel },
    ...(envConfig?.fabList.map(f => ({ value: f.id, label: f.name ?? f.id })) ?? []),
  ];

  const handleClose = () => {
    dispatch({ type: 'CLOSE_NEW_SCHEDULE_DIALOG' });
    setTab('form');
    setSchedFile(null);
    setEnvFile(null);
    setEntries([makeEntry(planStart)]);
  };

  // ── Upload tab ─────────────────────────────────────────────────────────────

  const handleImport = async () => {
    if (!schedFile && !envFile) return;
    setUploading(true);
    try {
      const [schedText, envText] = await Promise.all([
        schedFile ? readText(schedFile) : null,
        envFile ? readText(envFile) : null,
      ]);
      dispatch({
        type: 'MERGE_DATA',
        payload: {
          schedule: schedText ? parseScheduleYaml(schedText) : undefined,
          envConfig: envText ? parseEnvConfigYaml(envText) : undefined,
        },
      });
      handleClose();
    } catch (err) {
      dispatch({ type: 'SET_ERROR', payload: UI.mergeErrorMessage((err as Error).message) });
    } finally {
      setUploading(false);
    }
  };

  // ── Form tab ───────────────────────────────────────────────────────────────

  const addEntry = () =>
    setEntries(prev => [...prev, makeEntry(planStart)]);

  const removeEntry = (key: string) =>
    setEntries(prev => prev.filter(e => e.key !== key));

  const patchEntry = (key: string, patch: Partial<WTEntry>) =>
    setEntries(prev => prev.map(e => e.key === key ? { ...e, ...patch } : e));

  const handleWorkflowChange = (key: string, workflowId: string) => {
    const wf = validWorkflows.find(w => w.id === workflowId);
    patchEntry(key, {
      workflowId,
      opEntries: wf ? initOpEntries(wf) : [],
      phaseEndDates: wf ? wf.phaseList.map(() => '') : [],
    });
  };

  const setPhaseEndDate = (key: string, pi: number, val: string) =>
    setEntries(prev => prev.map(e => {
      if (e.key !== key) return e;
      const next = [...e.phaseEndDates];
      next[pi] = val;
      return { ...e, phaseEndDates: next };
    }));

  const setOpField = (key: string, pi: number, oi: number, field: keyof OpEntry, val: number) =>
    setEntries(prev => prev.map(e => {
      if (e.key !== key) return e;
      const next = e.opEntries.map((row, r) =>
        r === pi ? row.map((entry, c) => c === oi ? { ...entry, [field]: val } : entry) : row
      );
      return { ...e, opEntries: next };
    }));

  const handleFormOk = () => {
    if (!envConfig) return;
    const tasks: WorkflowTask[] = entries
      .filter(e => e.name.trim() && e.workflowId && e.startDate)
      .map(e => buildWorkflowTask(e, validWorkflows.find(w => w.id === e.workflowId)!));
    if (tasks.length === 0) return;
    dispatch({ type: 'ADD_WORKFLOW_TASKS', payload: tasks });
    handleClose();
  };

  const canUpload = (schedFile || envFile) && !uploading;
  const canFormSubmit = entries.some(e => e.name.trim() && e.workflowId && e.startDate);

  const S = styles;

  return (
    <div style={S.overlay} onClick={e => e.target === e.currentTarget && handleClose()}>
      <div style={S.modal}>
        <div style={S.titleBar}>{UI.newScheduleDialogTitle}</div>

        <div style={S.tabBar}>
          {(['form', 'upload'] as const).map(t => (
            <button key={t} style={tab === t ? S.tabActive : S.tabInactive} onClick={() => setTab(t)}>
              {t === 'upload' ? UI.tabImportLabel : UI.tabFormLabel}
            </button>
          ))}
        </div>

        <div style={S.body}>

          {/* ── UPLOAD TAB ── */}
          {tab === 'upload' && (
            <>
              <div style={S.hint}>
                {UI.mergeHintLine1}<br />
                {UI.mergeHintLine2}
              </div>
              <FileRow label={UI.scheduleFileMergeLabel} file={schedFile}
                onPick={() => schedRef.current?.click()} onClear={() => setSchedFile(null)} />
              <input ref={schedRef} type="file" accept=".yaml,.yml" style={{ display: 'none' }}
                onChange={e => setSchedFile(e.target.files?.[0] ?? null)} />
              <FileRow label={UI.envFileMergeLabel} file={envFile}
                onPick={() => envRef.current?.click()} onClear={() => setEnvFile(null)} />
              <input ref={envRef} type="file" accept=".yaml,.yml" style={{ display: 'none' }}
                onChange={e => setEnvFile(e.target.files?.[0] ?? null)} />
            </>
          )}

          {/* ── FORM TAB ── */}
          {tab === 'form' && (
            <>
              {entries.map((entry, idx) => {
                const wf = validWorkflows.find(w => w.id === entry.workflowId);
                return (
                  <div key={entry.key} style={S.card}>
                    <div style={S.cardHeader}>
                      <span style={S.cardTitle}>{UI.seibanEntryTitle(idx + 1, entry.name)}</span>
                      <div style={{ display: 'flex', gap: 4 }}>
                        <button style={S.iconBtn}
                          onClick={() => patchEntry(entry.key, { collapsed: !entry.collapsed })}
                          title={entry.collapsed ? UI.expandTitle : UI.collapseTitle}>
                          {entry.collapsed ? '▼' : '▲'}
                        </button>
                        <button style={{ ...S.iconBtn, color: '#b71c1c' }}
                          onClick={() => removeEntry(entry.key)} title={UI.deleteButton}>−</button>
                      </div>
                    </div>

                    {!entry.collapsed && (
                      <div style={S.cardBody}>
                        <Field label={UI.seibanNameRequiredLabel}>
                          <input style={S.input} value={entry.name} placeholder={UI.seibanNamePlaceholder}
                            onChange={e => patchEntry(entry.key, { name: e.target.value })} />
                        </Field>

                        <Field label={UI.workflowRequiredLabel}>
                          <SearchableSelect
                            value={entry.workflowId}
                            options={workflowOptions}
                            onChange={v => handleWorkflowChange(entry.key, v)}
                            placeholder={UI.selectPlaceholder}
                          />
                        </Field>

                        <Field label={UI.fabOptionalLabel}>
                          <SearchableSelect
                            value={entry.fabId}
                            options={fabOptions}
                            onChange={v => patchEntry(entry.key, { fabId: v })}
                            placeholder={UI.noneOptionLabel}
                          />
                        </Field>

                        <Field label={UI.workStartDateRequiredLabel}>
                          <input style={S.input} type="date" value={entry.startDate}
                            onChange={e => patchEntry(entry.key, { startDate: e.target.value })} />
                        </Field>

                        {/* 工程別設定 */}
                        {wf && entry.opEntries.length > 0 && (
                          <div>
                            <div style={S.sectionTitle}>{UI.phaseSettingsSectionTitle}</div>
                            {wf.phaseList.map((phase, pi) => (
                              <div key={phase.id} style={S.phaseBlock}>
                                <div style={S.phaseLabel}>{phase.name ?? phase.id}</div>

                                {/* Per-phase end date */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                  <span style={{ fontSize: 11, color: '#555', width: 68, flexShrink: 0 }}>{UI.phaseEndDateLabel}</span>
                                  <input
                                    style={{ ...S.input, flex: 1 }}
                                    type="date"
                                    value={entry.phaseEndDates[pi] ?? ''}
                                    onChange={e => setPhaseEndDate(entry.key, pi, e.target.value)}
                                  />
                                </div>

                                {/* Per-operation rows */}
                                <div style={S.opTableHeader}>
                                  <span style={{ flex: 1 }}>{UI.phaseColumnLabel}</span>
                                  <span style={{ width: 56, textAlign: 'center' }}>{UI.minWorkerLabel}</span>
                                  <span style={{ width: 56, textAlign: 'center' }}>{UI.maxWorkerLabel}</span>
                                  <span style={{ width: 56, textAlign: 'center' }}>{UI.workloadHoursLabelCompact}</span>
                                </div>
                                {phase.operationList.map((op, oi) => {
                                  const oe = entry.opEntries[pi]?.[oi];
                                  if (!oe) return null;
                                  return (
                                    <div key={op.id} style={S.opRow}>
                                      <span style={S.opName}>{op.name ?? op.id}</span>
                                      <input type="number" min={1} value={oe.minWorker}
                                        style={S.numInput}
                                        onChange={e => setOpField(entry.key, pi, oi, 'minWorker', Number(e.target.value))} />
                                      <input type="number" min={oe.minWorker} value={oe.maxWorker}
                                        style={S.numInput}
                                        onChange={e => setOpField(entry.key, pi, oi, 'maxWorker', Number(e.target.value))} />
                                      <input type="number" min={1} value={oe.workloadHours}
                                        style={S.numInput}
                                        onChange={e => setOpField(entry.key, pi, oi, 'workloadHours', Number(e.target.value))} />
                                    </div>
                                  );
                                })}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              <button style={S.addEntryBtn} onClick={addEntry}>{UI.addSeibanEntryBtn}</button>
            </>
          )}
        </div>

        <div style={S.footer}>
          {tab === 'upload' ? (
            <button style={canUpload ? S.primaryBtn : S.disabledBtn} onClick={handleImport} disabled={!canUpload}>
              {uploading ? UI.loadingLabel : UI.importBtn}
            </button>
          ) : (
            <button style={canFormSubmit ? S.primaryBtn : S.disabledBtn} onClick={handleFormOk} disabled={!canFormSubmit}>
              {UI.dialogOk}
            </button>
          )}
          <button style={S.cancelBtn} onClick={handleClose}>{UI.dialogCancel}</button>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function FileRow({ label, file, onPick, onClear }: {
  label: string; file: File | null; onPick: () => void; onClear: () => void;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={styles.fieldLabel}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button style={styles.chooseBtn} onClick={onPick}>{UI.chooseFile}</button>
        <span style={{ fontSize: 12, color: file ? '#333' : '#aaa' }}>
          {file ? file.name : UI.fileNotChosenShort}
        </span>
        {file && (
          <button onClick={onClear}
            style={{ padding: '1px 6px', border: '1px solid #ccc', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}>
            ×
          </button>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <label style={{ display: 'block', fontSize: 11, color: '#555', fontWeight: 'bold', marginBottom: 3 }}>
        {label}
      </label>
      {children}
    </div>
  );
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function readText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(UI.fileReadErrorMessage(file.name)));
    reader.readAsText(file, 'utf-8');
  });
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  overlay: {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 900,
  } as React.CSSProperties,
  modal: {
    backgroundColor: '#fff', borderRadius: 6, width: 560,
    maxHeight: '90vh', display: 'flex', flexDirection: 'column',
    boxShadow: '0 8px 28px rgba(0,0,0,0.35)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  } as React.CSSProperties,
  titleBar: {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold', flexShrink: 0,
  } as React.CSSProperties,
  tabBar: {
    display: 'flex', borderBottom: '1px solid #ddd',
    backgroundColor: '#f5f5f5', flexShrink: 0,
  } as React.CSSProperties,
  tabActive: {
    padding: '8px 18px', border: 'none', cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace', backgroundColor: '#fff',
    borderBottom: '2px solid #1976d2', color: '#1976d2', fontWeight: 'bold',
  } as React.CSSProperties,
  tabInactive: {
    padding: '8px 18px', border: 'none', cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace', backgroundColor: 'transparent', color: '#666',
  } as React.CSSProperties,
  body: { padding: 16, flex: 1, overflowY: 'auto' } as React.CSSProperties,
  footer: {
    display: 'flex', justifyContent: 'flex-end', gap: 8,
    padding: '12px 16px', borderTop: '1px solid #e0e0e0',
    backgroundColor: '#fafafa', flexShrink: 0,
  } as React.CSSProperties,
  hint: {
    fontSize: 11, color: '#555', backgroundColor: '#f0f4f8',
    border: '1px solid #c8d8e8', borderRadius: 3, padding: '7px 10px',
    marginBottom: 16, lineHeight: 1.6,
  } as React.CSSProperties,
  fieldLabel: { fontSize: 11, color: '#444', fontWeight: 'bold', marginBottom: 6 } as React.CSSProperties,
  chooseBtn: {
    padding: '4px 12px', border: '1px solid #aaa', borderRadius: 3,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#f5f5f5',
    fontFamily: 'MS Gothic, monospace', whiteSpace: 'nowrap',
  } as React.CSSProperties,
  card: {
    border: '1px solid #d0d5dd', borderRadius: 4, marginBottom: 10,
    backgroundColor: '#fafbfc', overflow: 'hidden',
  } as React.CSSProperties,
  cardHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '6px 10px', backgroundColor: '#e8eef6', borderBottom: '1px solid #d0d5dd',
  } as React.CSSProperties,
  cardTitle: { fontSize: 12, fontWeight: 'bold', color: '#1c2b3a' } as React.CSSProperties,
  cardBody: { padding: '10px 12px' } as React.CSSProperties,
  iconBtn: {
    padding: '2px 8px', border: '1px solid #ccc', borderRadius: 3,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#fff',
    fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  input: {
    width: '100%', padding: '4px 6px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 12, boxSizing: 'border-box', fontFamily: 'MS Gothic, monospace',
  } as React.CSSProperties,
  sectionTitle: {
    fontSize: 11, fontWeight: 'bold', color: '#444',
    borderBottom: '1px solid #e0e0e0', paddingBottom: 3, marginBottom: 8, marginTop: 4,
  } as React.CSSProperties,
  phaseBlock: {
    marginBottom: 12, padding: '8px 10px', background: '#f2f6fc',
    borderRadius: 3, border: '1px solid #d8e4f0',
  } as React.CSSProperties,
  phaseLabel: {
    fontSize: 11, color: '#1565c0', fontWeight: 'bold', marginBottom: 6,
  } as React.CSSProperties,
  opTableHeader: {
    display: 'flex', gap: 4, fontSize: 10, color: '#888', fontWeight: 'bold',
    marginBottom: 3, paddingLeft: 4,
  } as React.CSSProperties,
  opRow: {
    display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3,
  } as React.CSSProperties,
  opName: { flex: 1, fontSize: 11, color: '#444', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } as React.CSSProperties,
  numInput: {
    width: 56, padding: '2px 4px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 12, textAlign: 'right', fontFamily: 'MS Gothic, monospace', flexShrink: 0,
  } as React.CSSProperties,
  addEntryBtn: {
    width: '100%', padding: 8, border: '2px dashed #ccc', borderRadius: 4,
    backgroundColor: 'transparent', cursor: 'pointer', fontSize: 12,
    fontFamily: 'MS Gothic, monospace', color: '#666', marginTop: 4,
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
