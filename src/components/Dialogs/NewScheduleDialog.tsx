import { useState, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { parseEnvConfigYaml, parseScheduleYaml } from '../../services/yamlService';
import { WorkflowTask, PhaseTask, OperationTask } from '../../types/schedule';
import { Workflow } from '../../types/envConfig';

// ── Per-製番 form entry ───────────────────────────────────────────────────────

interface WTEntry {
  key: string;
  name: string;
  workflowId: string;
  fabId: string;
  startDate: string;
  endDate: string;
  collapsed: boolean;
  opHours: number[][];  // [phaseIdx][opIdx] = total workload hours
}

function makeEntry(startDate: string): WTEntry {
  return {
    key: `e_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
    name: '', workflowId: '', fabId: '',
    startDate, endDate: '',
    collapsed: false, opHours: [],
  };
}

function initOpHours(workflow: Workflow): number[][] {
  return workflow.phaseList.map(phase =>
    phase.operationList.map(op => op.workHours?.[0] ?? 8)
  );
}

function buildWorkflowTask(entry: WTEntry, workflow: Workflow): WorkflowTask {
  const base = entry.name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase().slice(0, 12)
    + '_' + Date.now().toString(36).slice(-4);
  return {
    id: base,
    name: entry.name,
    workflow: entry.workflowId,
    fab: entry.fabId,
    phaseTaskList: workflow.phaseList.map((phase, pi): PhaseTask => {
      const ptId = `${base}_p${pi}`;
      return {
        id: ptId,
        name: phase.name,
        phase: phase.id,
        startDate: entry.startDate,
        endDate: entry.endDate,
        operationTaskList: phase.operationList.map((op, oi): OperationTask => ({
          id: `${ptId}_o${oi}`,
          name: op.name,
          operation: op.id,
          workloadHours: entry.opHours[pi]?.[oi] ?? 8,
        })),
      };
    }),
  };
}

// ── Main component ────────────────────────────────────────────────────────────

export function NewScheduleDialog() {
  const { state, dispatch } = useAppContext();
  const { envConfig, schedule } = state;

  const [tab, setTab] = useState<'upload' | 'form'>('upload');

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

  const handleClose = () => {
    dispatch({ type: 'CLOSE_NEW_SCHEDULE_DIALOG' });
    setTab('upload');
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
      dispatch({ type: 'SET_ERROR', payload: `マージエラー: ${(err as Error).message}` });
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
    const wf = envConfig?.workflowList.find(w => w.id === workflowId);
    patchEntry(key, { workflowId, opHours: wf ? initOpHours(wf) : [] });
  };

  const setOpHour = (key: string, pi: number, oi: number, val: number) =>
    setEntries(prev => prev.map(e => {
      if (e.key !== key) return e;
      const next = e.opHours.map((row, r) =>
        r === pi ? row.map((h, c) => c === oi ? val : h) : row
      );
      return { ...e, opHours: next };
    }));

  const handleFormOk = () => {
    if (!envConfig) return;
    const tasks: WorkflowTask[] = entries
      .filter(e => e.name.trim() && e.workflowId && e.startDate && e.endDate)
      .map(e => buildWorkflowTask(e, envConfig.workflowList.find(w => w.id === e.workflowId)!));
    if (tasks.length === 0) return;
    dispatch({ type: 'ADD_WORKFLOW_TASKS', payload: tasks });
    handleClose();
  };

  const canUpload = (schedFile || envFile) && !uploading;
  const canFormSubmit = entries.some(e => e.name.trim() && e.workflowId && e.startDate && e.endDate);

  // ── Styles ─────────────────────────────────────────────────────────────────

  const S = styles;

  return (
    <div style={S.overlay} onClick={e => e.target === e.currentTarget && handleClose()}>
      <div style={S.modal}>
        {/* Title */}
        <div style={S.titleBar}>新規製番追加</div>

        {/* Tab bar */}
        <div style={S.tabBar}>
          {(['upload', 'form'] as const).map(t => (
            <button
              key={t}
              style={tab === t ? S.tabActive : S.tabInactive}
              onClick={() => setTab(t)}
            >
              {t === 'upload' ? 'ファイルからインポート' : 'フォームで追加'}
            </button>
          ))}
        </div>

        {/* Body */}
        <div style={S.body}>

          {/* ── UPLOAD TAB ── */}
          {tab === 'upload' && (
            <>
              <div style={S.hint}>
                既存データにマージします。同じIDの製番・作業者・Fabは無視されます。<br />
                どちらか一方だけでも読み込み可能です。
              </div>

              <FileRow
                label="Schedule.yaml（製番・割付を追加）"
                file={schedFile}
                onPick={() => schedRef.current?.click()}
                onClear={() => setSchedFile(null)}
              />
              <input ref={schedRef} type="file" accept=".yaml,.yml" style={{ display: 'none' }}
                onChange={e => setSchedFile(e.target.files?.[0] ?? null)} />

              <FileRow
                label="EnvConfig.yaml（作業者・Fab等を追加）"
                file={envFile}
                onPick={() => envRef.current?.click()}
                onClear={() => setEnvFile(null)}
              />
              <input ref={envRef} type="file" accept=".yaml,.yml" style={{ display: 'none' }}
                onChange={e => setEnvFile(e.target.files?.[0] ?? null)} />
            </>
          )}

          {/* ── FORM TAB ── */}
          {tab === 'form' && (
            <>
              {entries.map((entry, idx) => {
                const wf = envConfig?.workflowList.find(w => w.id === entry.workflowId);
                return (
                  <div key={entry.key} style={S.card}>
                    {/* Card header */}
                    <div style={S.cardHeader}>
                      <span style={S.cardTitle}>
                        製番 {idx + 1}{entry.name ? `：${entry.name}` : ''}
                      </span>
                      <div style={{ display: 'flex', gap: 4 }}>
                        <button
                          style={S.iconBtn}
                          onClick={() => patchEntry(entry.key, { collapsed: !entry.collapsed })}
                          title={entry.collapsed ? '展開' : '折りたたむ'}
                        >
                          {entry.collapsed ? '▼' : '▲'}
                        </button>
                        <button
                          style={{ ...S.iconBtn, color: '#b71c1c' }}
                          onClick={() => removeEntry(entry.key)}
                          title="削除"
                        >
                          −
                        </button>
                      </div>
                    </div>

                    {!entry.collapsed && (
                      <div style={S.cardBody}>
                        {/* 製番名 */}
                        <Field label="製番名 *">
                          <input style={S.input} value={entry.name}
                            placeholder="例: SU 1002B"
                            onChange={e => patchEntry(entry.key, { name: e.target.value })} />
                        </Field>

                        {/* ワークフロー */}
                        <Field label="ワークフロー *">
                          <select style={S.input} value={entry.workflowId}
                            onChange={e => handleWorkflowChange(entry.key, e.target.value)}>
                            <option value="">--- 選択 ---</option>
                            {envConfig?.workflowList.map(w => (
                              <option key={w.id} value={w.id}>{w.name ?? w.id}</option>
                            ))}
                          </select>
                        </Field>

                        {/* Fab */}
                        <Field label="Fab">
                          <select style={S.input} value={entry.fabId}
                            onChange={e => patchEntry(entry.key, { fabId: e.target.value })}>
                            <option value="">--- 選択 ---</option>
                            {envConfig?.fabList.map(f => (
                              <option key={f.id} value={f.id}>{f.name ?? f.id}</option>
                            ))}
                          </select>
                        </Field>

                        {/* 期間 */}
                        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                          <div style={{ flex: 1 }}>
                            <Field label="開始日 *">
                              <input style={S.input} type="date" value={entry.startDate}
                                onChange={e => patchEntry(entry.key, { startDate: e.target.value })} />
                            </Field>
                          </div>
                          <div style={{ flex: 1 }}>
                            <Field label="終了日 *">
                              <input style={S.input} type="date" value={entry.endDate}
                                onChange={e => patchEntry(entry.key, { endDate: e.target.value })} />
                            </Field>
                          </div>
                        </div>

                        {/* 工程別工数 */}
                        {wf && entry.opHours.length > 0 && (
                          <div>
                            <div style={S.hoursTitle}>工程別 作業工数</div>
                            {wf.phaseList.map((phase, pi) => (
                              <div key={phase.id} style={{ marginBottom: 8 }}>
                                <div style={S.phaseLabel}>{phase.name ?? phase.id}</div>
                                {phase.operationList.map((op, oi) => (
                                  <div key={op.id} style={S.opRow}>
                                    <span style={S.opName}>{op.name ?? op.id}</span>
                                    <input
                                      style={S.hoursInput}
                                      type="number" min={1}
                                      value={entry.opHours[pi]?.[oi] ?? 8}
                                      onChange={e => setOpHour(entry.key, pi, oi, Number(e.target.value))}
                                    />
                                    <span style={{ fontSize: 11, color: '#888' }}>h</span>
                                  </div>
                                ))}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              <button style={S.addEntryBtn} onClick={addEntry}>+ 製番追加</button>
            </>
          )}
        </div>

        {/* Footer */}
        <div style={S.footer}>
          {tab === 'upload' ? (
            <button
              style={canUpload ? S.primaryBtn : S.disabledBtn}
              onClick={handleImport} disabled={!canUpload}
            >
              {uploading ? '読み込み中...' : 'インポート'}
            </button>
          ) : (
            <button
              style={canFormSubmit ? S.primaryBtn : S.disabledBtn}
              onClick={handleFormOk} disabled={!canFormSubmit}
            >
              OK
            </button>
          )}
          <button style={S.cancelBtn} onClick={handleClose}>キャンセル</button>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function FileRow({ label, file, onPick, onClear }: {
  label: string; file: File | null;
  onPick: () => void; onClear: () => void;
}) {
  const S = styles;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={S.fieldLabel}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button style={S.chooseBtn} onClick={onPick}>ファイル選択</button>
        <span style={{ fontSize: 12, color: file ? '#333' : '#aaa' }}>
          {file ? file.name : '選択なし'}
        </span>
        {file && (
          <button
            onClick={onClear}
            style={{ padding: '1px 6px', border: '1px solid #ccc', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}
          >
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
    reader.onerror = () => reject(new Error(`読み込み失敗: ${file.name}`));
    reader.readAsText(file, 'utf-8');
  });
}

// ── Style constants ───────────────────────────────────────────────────────────

const styles = {
  overlay: {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 900,
  } as React.CSSProperties,
  modal: {
    backgroundColor: '#fff', borderRadius: 6, width: 540,
    maxHeight: '88vh', display: 'flex', flexDirection: 'column',
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
  body: {
    padding: 16, flex: 1, overflowY: 'auto',
  } as React.CSSProperties,
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
  fieldLabel: {
    fontSize: 11, color: '#444', fontWeight: 'bold', marginBottom: 6,
  } as React.CSSProperties,
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
  cardTitle: {
    fontSize: 12, fontWeight: 'bold', color: '#1c2b3a',
  } as React.CSSProperties,
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
  hoursTitle: {
    fontSize: 11, fontWeight: 'bold', color: '#444',
    borderBottom: '1px solid #e0e0e0', paddingBottom: 3, marginBottom: 6,
  } as React.CSSProperties,
  phaseLabel: {
    fontSize: 11, color: '#1565c0', fontWeight: 'bold',
    padding: '2px 0', marginBottom: 4,
  } as React.CSSProperties,
  opRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    marginBottom: 4, paddingLeft: 10,
  } as React.CSSProperties,
  opName: { fontSize: 11, color: '#444', width: 100, flexShrink: 0 } as React.CSSProperties,
  hoursInput: {
    width: 64, padding: '2px 6px', border: '1px solid #ccc', borderRadius: 3,
    fontSize: 12, textAlign: 'right', fontFamily: 'MS Gothic, monospace',
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
