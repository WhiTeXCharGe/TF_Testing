import { useMemo, useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { diffDays } from '../../utils/dateUtils';
import {
  buildModuleViewModel,
  ModuleNode,
  ModulePhase,
  ModuleTask,
} from './moduleViewModel';

interface Props {
  dates: string[];
}

const HEADER_H = 26;
const ROW_H = 34;
const CELL_W = 22;
const LEFT_W = 170;
const PANEL_W = 320;
const DOW_JA = ['日', '月', '火', '水', '木', '金', '土'];

type Selection =
  | { kind: 'koutei'; moduleId: string; phaseId: string }
  | { kind: 'task'; moduleId: string; phaseId: string; taskId: string }
  | null;

interface KouteiRow {
  type: 'koutei';
  key: string;
  module: ModuleNode;
  expanded: boolean;
}
interface TaskLineRow {
  type: 'taskline';
  key: string;
  module: ModuleNode;
  taskIndex: number;
}
type Row = KouteiRow | TaskLineRow;

function barGeom(
  start: string | null,
  end: string | null,
  viewStart: string,
  viewEnd: string,
): { left: number; width: number } | null {
  if (!start || !end) return null;
  const s = start < viewStart ? viewStart : start;
  const e = end > viewEnd ? viewEnd : end;
  if (e < s) return null;
  const left = diffDays(viewStart, s) * CELL_W;
  const width = (diffDays(s, e) + 1) * CELL_W;
  return { left, width };
}

export function DeviceViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig } = state;

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selection, setSelection] = useState<Selection>(null);

  const model = useMemo(() => {
    if (!schedule || !envConfig || dates.length === 0) {
      return { modules: [], monthGroups: [] };
    }
    return buildModuleViewModel(envConfig, schedule, dates);
  }, [schedule, envConfig, dates]);

  const viewStart = dates[0];
  const viewEnd = dates[dates.length - 1];
  const timelineWidth = dates.length * CELL_W;

  const toggle = (id: string) =>
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const m of model.modules) {
      const isExp = expanded.has(m.moduleId);
      out.push({ type: 'koutei', key: `k_${m.moduleId}`, module: m, expanded: isExp });
      if (isExp) {
        const maxTasks = m.phases.reduce((mx, p) => Math.max(mx, p.tasks.length), 0);
        for (let ti = 0; ti < maxTasks; ti++) {
          out.push({ type: 'taskline', key: `t_${m.moduleId}_${ti}`, module: m, taskIndex: ti });
        }
      }
    }
    return out;
  }, [model.modules, expanded]);

  const selectedPhase = useMemo<{ phase: ModulePhase } | null>(() => {
    if (selection?.kind !== 'koutei') return null;
    const m = model.modules.find(x => x.moduleId === selection.moduleId);
    const phase = m?.phases.find(p => p.phaseId === selection.phaseId);
    return phase ? { phase } : null;
  }, [selection, model.modules]);

  const selectedTask = useMemo<{ task: ModuleTask; phase: ModulePhase } | null>(() => {
    if (selection?.kind !== 'task') return null;
    const m = model.modules.find(x => x.moduleId === selection.moduleId);
    const phase = m?.phases.find(p => p.phaseId === selection.phaseId);
    const task = phase?.tasks.find(t => t.taskId === selection.taskId);
    return task && phase ? { task, phase } : null;
  }, [selection, model.modules]);

  if (!schedule || !envConfig || dates.length === 0) {
    return <div style={{ flex: 1, background: '#f8f9fa' }} />;
  }

  const closePanel = () => setSelection(null);

  const renderBar = (
    geom: { left: number; width: number },
    label: string,
    color: string,
    isSel: boolean,
    bold: boolean,
    onClick: () => void,
    title: string,
  ) => (
    <div
      onClick={e => {
        e.stopPropagation();
        onClick();
      }}
      title={title}
      style={{
        position: 'absolute',
        left: geom.left + 1,
        top: 5,
        width: Math.max(CELL_W - 2, geom.width - 2),
        height: ROW_H - 10,
        background: color,
        borderRadius: 4,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        paddingLeft: 6,
        paddingRight: 6,
        fontSize: 11,
        color: '#1f2d3d',
        fontWeight: bold ? 700 : 500,
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
        outline: isSel ? '2px solid #1565c0' : 'none',
        outlineOffset: 1,
        boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
      }}
    >
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
    </div>
  );

  return (
    <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden', background: '#fff' }}>
      {/* ── LEFT: seiban labels ───────────────────────────── */}
      <div style={{ width: LEFT_W, flexShrink: 0, borderRight: '1px solid #c9d5e3', display: 'flex', flexDirection: 'column', background: '#f7fafc', overflow: 'hidden' }}>
        <div style={{ height: HEADER_H * 3, flexShrink: 0, borderBottom: '1px solid #c9d5e3', background: '#f2f6fb', display: 'flex', alignItems: 'center', padding: '0 10px', fontWeight: 700, fontSize: 12, color: '#1e334b' }}>
          {UI.deviceCodeLabel}
        </div>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {rows.map(row => (
            <div
              key={row.key}
              onClick={() => {
                if (row.type === 'koutei') {
                  closePanel();
                  toggle(row.module.moduleId);
                } else {
                  closePanel();
                }
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                height: ROW_H,
                paddingLeft: 8,
                paddingRight: 8,
                borderBottom: '1px solid #ecf1f7',
                background: row.type === 'koutei' ? '#e8eef5' : '#ffffff',
                cursor: 'pointer',
                fontWeight: row.type === 'koutei' ? 700 : 400,
                fontSize: 12,
                color: '#25384f',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              title={row.type === 'koutei' ? row.module.moduleName : ''}
            >
              {row.type === 'koutei' && (
                <>
                  <span style={{ marginRight: 5, fontSize: 12, color: '#5a7fa0', flexShrink: 0, fontWeight: 700 }}>
                    {row.expanded ? '−' : '+'}
                  </span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{row.module.moduleName}</span>
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── CENTER: timeline ──────────────────────────────── */}
      <div style={{ flex: 1, minWidth: 0, overflowX: 'auto', overflowY: 'hidden' }} onClick={closePanel}>
        <div style={{ minWidth: timelineWidth }}>
          {/* Header: month / day / dow */}
          <div style={{ background: '#f4f8fc', borderBottom: '1px solid #c9d5e3' }}>
            <div style={{ display: 'flex', height: HEADER_H }}>
              {model.monthGroups.map(g => (
                <div key={`mg_${g.startIndex}`} style={{ width: g.span * CELL_W, minWidth: g.span * CELL_W, borderRight: '1px solid #d7e1ed', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 12, color: '#18324f' }}>
                  {g.label}
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', height: HEADER_H }}>
              {dates.map(d => {
                const [, , dd] = d.split('-');
                return (
                  <div key={`dd_${d}`} style={{ width: CELL_W, minWidth: CELL_W, borderRight: '1px solid #e4ebf4', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#2a3f56' }}>
                    {Number(dd)}
                  </div>
                );
              })}
            </div>
            <div style={{ display: 'flex', height: HEADER_H }}>
              {dates.map(d => {
                const dow = DOW_JA[new Date(`${d}T00:00:00`).getDay()] ?? '';
                const weekend = dow === '土' || dow === '日';
                return (
                  <div key={`dw_${d}`} style={{ width: CELL_W, minWidth: CELL_W, borderRight: '1px solid #e4ebf4', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: weekend ? '#b54747' : '#2a3f56', background: weekend ? '#fff5f5' : '#f8fbff' }}>
                    {dow}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Body rows */}
          <div>
            {rows.map(row => (
              <div
                key={row.key}
                style={{
                  position: 'relative',
                  height: ROW_H,
                  borderBottom: '1px solid #ecf1f7',
                  background: row.type === 'koutei' ? '#eef3f8' : '#ffffff',
                  minWidth: timelineWidth,
                }}
              >
                {/* weekend grid */}
                {dates.map((d, di) => {
                  const dow = new Date(`${d}T00:00:00`).getDay();
                  const weekend = dow === 0 || dow === 6;
                  return (
                    <div key={`g_${row.key}_${di}`} style={{ position: 'absolute', left: di * CELL_W, top: 0, width: CELL_W, height: ROW_H, borderRight: '1px solid #eef2f7', background: weekend ? 'rgba(181,71,71,0.05)' : undefined, pointerEvents: 'none' }} />
                  );
                })}

                {row.type === 'koutei'
                  ? row.module.phases.map(ph => {
                      const barStart = ph.barStartDate ?? ph.planStartDate;
                      const barEnd = ph.barEndDate ?? ph.planEndDate;
                      const geom = barGeom(barStart, barEnd, viewStart, viewEnd);
                      if (!geom) return null;
                      const isSel =
                        selection?.kind === 'koutei' &&
                        selection.moduleId === row.module.moduleId &&
                        selection.phaseId === ph.phaseId;
                      return (
                        <span key={`kb_${ph.phaseId}`}>
                          {renderBar(
                            geom,
                            ph.phaseName,
                            ph.color,
                            isSel,
                            true,
                            () => setSelection({ kind: 'koutei', moduleId: row.module.moduleId, phaseId: ph.phaseId }),
                            `${ph.phaseName}\n${barStart} 〜 ${barEnd}`,
                          )}
                        </span>
                      );
                    })
                  : row.module.phases.map(ph => {
                      const t = ph.tasks[row.taskIndex];
                      if (!t) return null;
                      const geom = barGeom(t.startDate, t.endDate, viewStart, viewEnd);
                      if (!geom) return null;
                      const isSel =
                        selection?.kind === 'task' &&
                        selection.moduleId === row.module.moduleId &&
                        selection.phaseId === ph.phaseId &&
                        selection.taskId === t.taskId;
                      return (
                        <span key={`tb_${ph.phaseId}_${t.taskId}`}>
                          {renderBar(
                            geom,
                            t.taskName,
                            t.color,
                            isSel,
                            false,
                            () => setSelection({ kind: 'task', moduleId: row.module.moduleId, phaseId: ph.phaseId, taskId: t.taskId }),
                            `${t.taskName}\n${t.startDate ?? ''} 〜 ${t.endDate ?? ''}`,
                          )}
                        </span>
                      );
                    })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── RIGHT: panel (only when a bar is selected) ────── */}
      {(selectedPhase || selectedTask) && (
        <div style={{ width: PANEL_W, flexShrink: 0, borderLeft: '1px solid #c9d5e3', background: '#fafafa', overflowY: 'auto', fontFamily: 'MS Gothic, monospace', fontSize: 12 }}>
          {selectedPhase ? (
            <KouteiPanel
              moduleId={(selection as { moduleId: string }).moduleId}
              phase={selectedPhase.phase}
              onChange={(updates) =>
                dispatch({
                  type: 'UPDATE_PHASE_TASK',
                  payload: { workflowTaskId: (selection as { moduleId: string }).moduleId, phaseTaskId: selectedPhase.phase.phaseId, updates },
                })
              }
            />
          ) : selectedTask ? (
            <TaskPanel
              task={selectedTask.task}
              phase={selectedTask.phase}
              workers={envConfig.workerList}
              onChangeWorker={(assignmentIndex, workerId) =>
                dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: assignmentIndex, updates: { worker: workerId } } })
              }
            />
          ) : null}
        </div>
      )}
    </div>
  );
}

// ── Right panel: koutei (phase) — editable plan dates ────
function KouteiPanel({
  phase,
  onChange,
}: {
  moduleId: string;
  phase: ModulePhase;
  onChange: (updates: { startDate?: string; endDate?: string }) => void;
}) {
  return (
    <div style={{ padding: 12 }}>
      <div style={panelTitle}>{phase.phaseName}</div>
      <Field label="作業開始可能日">
        <input
          type="date"
          value={phase.planStartDate}
          onChange={e => onChange({ startDate: e.target.value })}
          style={inputStyle}
        />
      </Field>
      <Field label="終了希望日">
        <input
          type="date"
          value={phase.planEndDate}
          onChange={e => onChange({ endDate: e.target.value })}
          style={inputStyle}
        />
      </Field>
      <Field label="作業期間">
        {phase.barStartDate && phase.barEndDate
          ? `${phase.barStartDate} 〜 ${phase.barEndDate}`
          : '—'}
      </Field>
      <Field label="割り当て作業者">{phase.workerCount}名</Field>
    </div>
  );
}

// ── Right panel: task — editable worker assignment ───────
function TaskPanel({
  task,
  phase,
  workers,
  onChangeWorker,
}: {
  task: ModuleTask;
  phase: ModulePhase;
  workers: { id: string; name?: string }[];
  onChangeWorker: (assignmentIndex: number, workerId: string) => void;
}) {
  return (
    <div style={{ padding: 12 }}>
      <div style={panelTitle}>{task.taskName}</div>
      <Field label="作業ID">{task.taskId}</Field>
      <Field label="工程">{phase.phaseName}</Field>
      <Field label="作業期間">
        {task.startDate && task.endDate ? `${task.startDate} 〜 ${task.endDate}` : '—'}
      </Field>
      <Field label="人数">
        {task.slots.length} / {task.maxWorker}（最小 {task.minWorker}）
      </Field>

      <div style={{ ...labelStyle, marginTop: 14, marginBottom: 6 }}>作業者割り当て</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {task.slots.length === 0 && (
          <div style={{ color: '#9aa8b8', fontStyle: 'italic' }}>割り当てなし</div>
        )}
        {task.slots.map(slot => (
          <div key={slot.assignmentIndex} style={{ padding: '8px', background: '#fff', border: '1px solid #e2e8f0', borderRadius: 4 }}>
            <div style={{ color: '#5a6b7d', fontSize: 11, marginBottom: 4 }}>
              {slot.startDate} 〜 {slot.endDate}
            </div>
            <select
              value={slot.workerId}
              onChange={e => onChangeWorker(slot.assignmentIndex, e.target.value)}
              style={{ ...inputStyle, width: '100%' }}
            >
              {workers.map(w => (
                <option key={w.id} value={w.id}>
                  {w.name ?? w.id}
                </option>
              ))}
            </select>
            <div style={{ color: '#5a6b7d', fontSize: 11, marginTop: 4 }}>{slot.companyName}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── helpers ──────────────────────────────────────────────
const panelTitle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 'bold',
  marginBottom: 12,
  color: '#1565c0',
  borderBottom: '1px solid #ccc',
  paddingBottom: 6,
};

const labelStyle: React.CSSProperties = { color: '#666', fontSize: 11, fontWeight: 700 };

const inputStyle: React.CSSProperties = {
  padding: '5px 6px',
  border: '1px solid #b8c6d5',
  borderRadius: 3,
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 10 }}>
      <span style={labelStyle}>{label}</span>
      <span style={{ color: '#222', fontWeight: 'bold' }}>{children}</span>
    </div>
  );
}