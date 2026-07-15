import { useMemo, useRef, useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { diffDays } from '../../utils/dateUtils';
import { buildModuleViewModel, ModuleNode, ModulePhase, ModuleTask } from './moduleViewModel';
import { SearchableSelect } from '../common/SearchableSelect';

interface Props { dates: string[] }

const HEADER_H = 26;
const ROW_H = 34;
const CELL_W = 22;
const MODULE_COL_W = 130;
const ATTR_COL_W = 90;
const LEFT_W = MODULE_COL_W + ATTR_COL_W;
const PANEL_W = 320;
const DOW_JA = ['日', '月', '火', '水', '木', '金', '土'];

type Selection =
  | { kind: 'koutei'; moduleId: string; phaseId: string }
  | { kind: 'task'; moduleId: string; phaseId: string; taskId: string }
  | null;

interface KouteiRow { type: 'koutei'; key: string; module: ModuleNode; expanded: boolean }
interface TaskLineRow { type: 'taskline'; key: string; module: ModuleNode; taskIndex: number }
type Row = KouteiRow | TaskLineRow;

function barGeom(start: string | null, end: string | null, viewStart: string, viewEnd: string) {
  if (!start || !end) return null;
  const s = start < viewStart ? viewStart : start;
  const e = end > viewEnd ? viewEnd : end;
  if (e < s) return null;
  return { left: diffDays(viewStart, s) * CELL_W, width: (diffDays(s, e) + 1) * CELL_W };
}

export function DeviceViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, moduleViewFilter } = state;

  const leftBodyRef = useRef<HTMLDivElement>(null);
  const rightScrollRef = useRef<HTMLDivElement>(null);

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selection, setSelection] = useState<Selection>(null);

  const model = useMemo(() => {
    if (!schedule || !envConfig || dates.length === 0) return { modules: [], monthGroups: [] };
    return buildModuleViewModel(envConfig, schedule, dates);
  }, [schedule, envConfig, dates]);

  // Apply module view global filter
  const filteredModules = useMemo(() => {
    const { workerIds, fabIds, regionIds } = moduleViewFilter;
    const noFilter = !workerIds.length && !fabIds.length && !regionIds.length;
    if (noFilter) return model.modules;

    const fabToRegion = new Map(envConfig?.fabList.map(f => [f.id, f.region ?? '']) ?? []);

    // Build workerIds per module from assignments
    const moduleWorkers = new Map<string, Set<string>>();
    if (schedule && workerIds.length > 0) {
      const opToModule = new Map<string, string>();
      for (const wt of schedule.workflowTaskList) {
        for (const pt of wt.phaseTaskList) {
          for (const ot of pt.operationTaskList) opToModule.set(ot.id, wt.id);
        }
      }
      for (const a of schedule.assignmentList) {
        const mid = opToModule.get(a.operationTask);
        if (!mid) continue;
        if (!moduleWorkers.has(mid)) moduleWorkers.set(mid, new Set());
        moduleWorkers.get(mid)!.add(a.worker);
      }
    }

    return model.modules.filter(m => {
      if (workerIds.length > 0) {
        const mw = moduleWorkers.get(m.moduleId);
        if (!mw || !workerIds.some(id => mw.has(id))) return false;
      }
      if (fabIds.length > 0) {
        if (!m.fab || !fabIds.includes(m.fab)) return false;
      }
      if (regionIds.length > 0) {
        const reg = m.region ?? (m.fab ? fabToRegion.get(m.fab) : undefined) ?? '';
        if (!regionIds.includes(reg)) return false;
      }
      return true;
    });
  }, [model.modules, moduleViewFilter, schedule, envConfig]);

  const viewStart = dates[0];
  const viewEnd = dates[dates.length - 1];
  const timelineWidth = dates.length * CELL_W;
  const totalRows = useMemo(() => {
    let n = 0;
    for (const m of filteredModules) {
      n += 1;
      if (expanded.has(m.moduleId)) {
        n += Math.max(...m.phases.map(p => p.tasks.length), 0);
      }
    }
    return n;
  }, [filteredModules, expanded]);

  const toggle = (id: string) =>
    setExpanded(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  const onScroll = () => {
    if (leftBodyRef.current && rightScrollRef.current)
      leftBodyRef.current.scrollTop = rightScrollRef.current.scrollTop;
  };

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const m of filteredModules) {
      const isExp = expanded.has(m.moduleId);
      out.push({ type: 'koutei', key: `k_${m.moduleId}`, module: m, expanded: isExp });
      if (isExp) {
        const maxT = m.phases.reduce((mx, p) => Math.max(mx, p.tasks.length), 0);
        for (let ti = 0; ti < maxT; ti++)
          out.push({ type: 'taskline', key: `t_${m.moduleId}_${ti}`, module: m, taskIndex: ti });
      }
    }
    return out;
  }, [filteredModules, expanded]);

  const selectedPhase = useMemo<{ phase: ModulePhase; module: ModuleNode } | null>(() => {
    if (selection?.kind !== 'koutei') return null;
    const m = filteredModules.find(x => x.moduleId === selection.moduleId);
    const phase = m?.phases.find(p => p.phaseId === selection.phaseId);
    return phase && m ? { phase, module: m } : null;
  }, [selection, filteredModules]);

  const selectedTask = useMemo<{ task: ModuleTask; phase: ModulePhase; module: ModuleNode } | null>(() => {
    if (selection?.kind !== 'task') return null;
    const m = filteredModules.find(x => x.moduleId === selection.moduleId);
    const phase = m?.phases.find(p => p.phaseId === selection.phaseId);
    const task = phase?.tasks.find(t => t.taskId === selection.taskId);
    return task && phase && m ? { task, phase, module: m } : null;
  }, [selection, filteredModules]);

  if (!schedule || !envConfig || dates.length === 0) return <div style={{ flex: 1, background: '#f8f9fa' }} />;

  const closePanel = () => setSelection(null);

  const renderBar = (
    geom: { left: number; width: number }, label: string, color: string,
    isSel: boolean, bold: boolean, onClick: () => void, title: string,
  ) => (
    <div
      onClick={e => { e.stopPropagation(); onClick(); }}
      title={title}
      style={{
        position: 'absolute', left: geom.left + 1, top: 5,
        width: Math.max(CELL_W - 2, geom.width - 2), height: ROW_H - 10,
        background: color, borderRadius: 4, cursor: 'pointer',
        display: 'flex', alignItems: 'center', paddingLeft: 6, paddingRight: 6,
        fontSize: 11, color: '#1f2d3d', fontWeight: bold ? 700 : 500,
        overflow: 'hidden', whiteSpace: 'nowrap', boxSizing: 'border-box',
        outline: isSel ? '2px solid #1565c0' : 'none', outlineOffset: 1,
        boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
      }}
    >
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
    </div>
  );

  const showPanel = !!(selectedPhase || selectedTask);

  return (
    <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden', background: '#fff' }}>

      {/* ── LEFT: seiban + workflow attribute labels ────────────────────────── */}
      <div style={{ width: LEFT_W, flexShrink: 0, borderRight: '1px solid #c9d5e3', display: 'flex', flexDirection: 'column', background: '#f7fafc', overflow: 'hidden' }}>
        {/* Sticky header placeholder */}
        <div style={{ height: HEADER_H * 3, flexShrink: 0, borderBottom: '1px solid #c9d5e3', background: '#f2f6fb', display: 'flex', alignItems: 'center' }}>
          <div style={{ width: MODULE_COL_W, flexShrink: 0, padding: '0 10px', fontWeight: 700, fontSize: 12, color: '#1e334b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {UI.deviceCodeLabel}
          </div>
          <div style={{ width: ATTR_COL_W, flexShrink: 0, borderLeft: '1px solid #c9d5e3', padding: '0 10px', fontWeight: 700, fontSize: 12, color: '#1e334b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {UI.deviceAttributeLabel}
          </div>
        </div>
        {/* Scrollable body — synced with right */}
        <div ref={leftBodyRef} style={{ flex: 1, overflowY: 'hidden' }}>
          <div style={{ minHeight: totalRows * ROW_H }}>
            {rows.map(row => (
              <div
                key={row.key}
                onClick={() => { closePanel(); if (row.type === 'koutei') toggle(row.module.moduleId); }}
                style={{
                  display: 'flex', alignItems: 'center', height: ROW_H,
                  borderBottom: '1px solid #ecf1f7',
                  background: row.type === 'koutei' ? '#e8eef5' : '#ffffff',
                  cursor: 'pointer', fontWeight: row.type === 'koutei' ? 700 : 400,
                  fontSize: 12, color: '#25384f',
                }}
              >
                <div
                  style={{
                    width: MODULE_COL_W, flexShrink: 0, paddingLeft: 8, paddingRight: 8,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    display: 'flex', alignItems: 'center',
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
                <div
                  style={{
                    width: ATTR_COL_W, flexShrink: 0, borderLeft: '1px solid #ecf1f7',
                    paddingLeft: 8, paddingRight: 8,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}
                  title={row.type === 'koutei' ? row.module.workflowName : ''}
                >
                  {row.type === 'koutei' ? row.module.workflowName : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── CENTER: timeline (horizontal+vertical scroll) ──────────────────── */}
      <div
        ref={rightScrollRef}
        onScroll={onScroll}
        style={{ flex: 1, minWidth: 0, overflow: 'auto', position: 'relative' }}
        onClick={closePanel}
      >
        <div style={{ minWidth: timelineWidth }}>
          {/* Sticky header: month / day / dow */}
          <div style={{ position: 'sticky', top: 0, zIndex: 4, background: '#f4f8fc', borderBottom: '1px solid #c9d5e3' }}>
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
            {rows.map((row, rowIdx) => (
              <div
                key={row.key}
                style={{
                  position: 'relative', height: ROW_H, borderBottom: '1px solid #ecf1f7',
                  background: row.type === 'koutei'
                    ? (rowIdx % 2 === 0 ? '#eef3f8' : '#e8f0f6')
                    : (rowIdx % 2 === 0 ? '#ffffff' : '#f9fbfd'),
                  minWidth: timelineWidth,
                }}
              >
                {/* Weekend column tints */}
                {dates.map((d, di) => {
                  const dow = new Date(`${d}T00:00:00`).getDay();
                  const weekend = dow === 0 || dow === 6;
                  return (
                    <div key={`g_${row.key}_${di}`} style={{ position: 'absolute', left: di * CELL_W, top: 0, width: CELL_W, height: ROW_H, borderRight: '1px solid #eef2f7', pointerEvents: 'none' }} />
                  );
                })}

                {row.type === 'koutei'
                  ? row.module.phases.map(ph => {
                      const geom = barGeom(ph.barStartDate ?? ph.planStartDate, ph.barEndDate ?? ph.planEndDate, viewStart, viewEnd);
                      if (!geom) return null;
                      const isSel = selection?.kind === 'koutei' && selection.moduleId === row.module.moduleId && selection.phaseId === ph.phaseId;
                      return (
                        <span key={`kb_${ph.phaseId}`}>
                          {renderBar(geom, ph.phaseName, ph.color, isSel, true,
                            () => setSelection({ kind: 'koutei', moduleId: row.module.moduleId, phaseId: ph.phaseId }),
                            `${ph.phaseName}\n${ph.barStartDate ?? ph.planStartDate} 〜 ${ph.barEndDate ?? ph.planEndDate}`)}
                        </span>
                      );
                    })
                  : row.module.phases.map(ph => {
                      const t = ph.tasks[row.taskIndex];
                      if (!t) return null;
                      const geom = barGeom(t.startDate, t.endDate, viewStart, viewEnd);
                      if (!geom) return null;
                      const isSel = selection?.kind === 'task' && selection.moduleId === row.module.moduleId && selection.phaseId === ph.phaseId && selection.taskId === t.taskId;
                      return (
                        <span key={`tb_${ph.phaseId}_${t.taskId}`}>
                          {renderBar(geom, t.taskName, t.color, isSel, false,
                            () => setSelection({ kind: 'task', moduleId: row.module.moduleId, phaseId: ph.phaseId, taskId: t.taskId }),
                            `${t.taskName}\n${t.startDate ?? ''} 〜 ${t.endDate ?? ''}`)}
                        </span>
                      );
                    })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── RIGHT: side panel ──────────────────────────────────────────────── */}
      {showPanel && (
        <div style={{ width: PANEL_W, flexShrink: 0, borderLeft: '1px solid #c9d5e3', background: '#fafafa', overflowY: 'auto', fontFamily: 'MS Gothic, monospace', fontSize: 12 }}>
          {selectedPhase ? (
            <KouteiPanel
              module={selectedPhase.module}
              phase={selectedPhase.phase}
              envConfig={envConfig}
              onChange={updates => dispatch({ type: 'UPDATE_PHASE_TASK', payload: { workflowTaskId: selectedPhase.module.moduleId, phaseTaskId: selectedPhase.phase.phaseId, updates } })}
            />
          ) : selectedTask ? (
            <TaskPanel
              task={selectedTask.task}
              phase={selectedTask.phase}
              module={selectedTask.module}
              envConfig={envConfig}
              onChangeWorker={(ai, wid) => dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: ai, updates: { worker: wid } } })}
              onChangeOpTask={(oid, updates) => dispatch({ type: 'UPDATE_OPERATION_TASK', payload: { workflowTaskId: selectedTask.module.moduleId, phaseTaskId: selectedTask.phase.phaseId, operationTaskId: oid, updates } })}
            />
          ) : null}
        </div>
      )}
    </div>
  );
}

// ── KouteiPanel ───────────────────────────────────────────────────────────────

function KouteiPanel({ module, phase, envConfig, onChange }: {
  module: ModuleNode;
  phase: ModulePhase;
  envConfig: import('../../types/envConfig').EnvConfig;
  onChange: (updates: { startDate?: string; endDate?: string; description?: string }) => void;
}) {
  const fab = module.fab ? envConfig.fabList.find(f => f.id === module.fab) : undefined;
  const regionId = module.region ?? fab?.region;
  const region = regionId ? envConfig.regionList.find(r => r.id === regionId) : undefined;
  const phaseKey = `${module.moduleId}_${phase.phaseId}`;

  return (
    <div style={{ padding: 12 }}>
      <div style={panelTitle}>{phase.phaseName}</div>

      <Field label="製番">{module.moduleName}</Field>

      {fab && <Field label="Fab">{fab.name ?? fab.id}</Field>}
      {region && <Field label="Region">{region.name ?? region.id}</Field>}

      <Field label="作業開始可能日">
        <input type="date" value={phase.planStartDate} onChange={e => onChange({ startDate: e.target.value })} style={inputStyle} />
      </Field>
      <Field label="終了希望日">
        <input type="date" value={phase.planEndDate} onChange={e => onChange({ endDate: e.target.value })} style={inputStyle} />
      </Field>
      <Field label="実績期間">
        {phase.barStartDate && phase.barEndDate ? `${phase.barStartDate} 〜 ${phase.barEndDate}` : '—'}
      </Field>
      <Field label="割り当て作業者">{phase.workerCount}名</Field>

      <div style={{ marginBottom: 10 }}>
        <span style={labelStyle}>備考</span>
        <textarea
          key={phaseKey}
          defaultValue={phase.description ?? ''}
          onBlur={e => onChange({ description: e.target.value })}
          rows={3}
          style={{ ...inputStyle, width: '100%', resize: 'vertical', fontFamily: 'MS Gothic, monospace', marginTop: 2 }}
          placeholder="備考を入力..."
        />
      </div>
    </div>
  );
}

// ── TaskPanel ─────────────────────────────────────────────────────────────────

function TaskPanel({ task, phase, module, envConfig, onChangeWorker, onChangeOpTask }: {
  task: ModuleTask;
  phase: ModulePhase;
  module: ModuleNode;
  envConfig: import('../../types/envConfig').EnvConfig;
  onChangeWorker: (assignmentIndex: number, workerId: string) => void;
  onChangeOpTask: (operationTaskId: string, updates: { recommendsWorkerMin?: number; recommendsWorkerMax?: number; workloadHours?: number; description?: string }) => void;
}) {
  const [minDraft, setMinDraft] = useState(task.minWorker);
  const [maxDraft, setMaxDraft] = useState(task.maxWorker);
  const [workloadDraft, setWorkloadDraft] = useState(task.workloadHours);

  const workerOptions = envConfig.workerList.map(w => {
    const co = envConfig.workerCompanyList.find(c => c.id === w.workerCompany);
    return { value: w.id, label: w.name ?? w.id, sub: co?.name ?? '' };
  });

  const commitMin = (v: number) => {
    const clamped = Math.max(1, v);
    setMinDraft(clamped);
    onChangeOpTask(task.taskId, { recommendsWorkerMin: clamped });
  };
  const commitMax = (v: number) => {
    const clamped = Math.max(minDraft, v);
    setMaxDraft(clamped);
    onChangeOpTask(task.taskId, { recommendsWorkerMax: clamped });
  };
  const commitWorkload = (v: number) => {
    const clamped = Math.max(1, v);
    setWorkloadDraft(clamped);
    onChangeOpTask(task.taskId, { workloadHours: clamped });
  };

  return (
    <div style={{ padding: 12 }}>
      <div style={panelTitle}>{task.taskName}</div>

      <Field label="製番">{module.moduleName}</Field>
      <Field label="工程">{phase.phaseName}</Field>
      <Field label="期間">{task.startDate && task.endDate ? `${task.startDate} 〜 ${task.endDate}` : '—'}</Field>

      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <span style={labelStyle}>最小人数</span>
          <input
            type="number" min={1} value={minDraft}
            onChange={e => setMinDraft(Number(e.target.value))}
            onBlur={e => commitMin(Number(e.target.value))}
            onKeyDown={e => { if (e.key === 'Enter') commitMin(Number((e.target as HTMLInputElement).value)); }}
            style={{ ...inputStyle, width: '100%' }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <span style={labelStyle}>最大人数</span>
          <input
            type="number" min={minDraft} value={maxDraft}
            onChange={e => setMaxDraft(Number(e.target.value))}
            onBlur={e => commitMax(Number(e.target.value))}
            onKeyDown={e => { if (e.key === 'Enter') commitMax(Number((e.target as HTMLInputElement).value)); }}
            style={{ ...inputStyle, width: '100%' }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <span style={labelStyle}>工数 (h)</span>
          <input
            type="number" min={1} value={workloadDraft}
            onChange={e => setWorkloadDraft(Number(e.target.value))}
            onBlur={e => commitWorkload(Number(e.target.value))}
            onKeyDown={e => { if (e.key === 'Enter') commitWorkload(Number((e.target as HTMLInputElement).value)); }}
            style={{ ...inputStyle, width: '100%' }}
          />
        </div>
      </div>

      <div style={{ ...labelStyle, marginBottom: 6 }}>
        作業者割り当て ({task.slots.length} / {task.maxWorker}名)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {task.slots.length === 0 && (
          <div style={{ color: '#9aa8b8', fontStyle: 'italic', fontSize: 11 }}>割り当てなし</div>
        )}
        {task.slots.map(slot => (
          <div key={slot.assignmentIndex} style={{ padding: 8, background: '#fff', border: '1px solid #e2e8f0', borderRadius: 4 }}>
            <div style={{ color: '#5a6b7d', fontSize: 10, marginBottom: 4 }}>{slot.startDate} 〜 {slot.endDate}</div>
            <SearchableSelect
              value={slot.workerId}
              options={workerOptions}
              onChange={v => onChangeWorker(slot.assignmentIndex, v)}
            />
            {slot.companyName && <div style={{ color: '#5a6b7d', fontSize: 10, marginTop: 3 }}>{slot.companyName}</div>}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12 }}>
        <span style={labelStyle}>備考</span>
        <textarea
          key={task.taskId}
          defaultValue={task.description ?? ''}
          onBlur={e => onChangeOpTask(task.taskId, { description: e.target.value })}
          rows={3}
          style={{ ...inputStyle, width: '100%', resize: 'vertical', fontFamily: 'MS Gothic, monospace', marginTop: 2 }}
          placeholder="備考を入力..."
        />
      </div>
    </div>
  );
}

// ── Shared helpers ────────────────────────────────────────────────────────────

const panelTitle: React.CSSProperties = {
  fontSize: 13, fontWeight: 'bold', marginBottom: 12, color: '#1565c0',
  borderBottom: '1px solid #ccc', paddingBottom: 6,
};
const labelStyle: React.CSSProperties = { color: '#666', fontSize: 11, fontWeight: 700, display: 'block', marginBottom: 2 };
const inputStyle: React.CSSProperties = { padding: '4px 6px', border: '1px solid #b8c6d5', borderRadius: 3, fontSize: 11, fontFamily: 'MS Gothic, monospace', boxSizing: 'border-box' };

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 10 }}>
      <span style={labelStyle}>{label}</span>
      <span style={{ color: '#222', fontWeight: 'bold', fontSize: 12 }}>{children}</span>
    </div>
  );
}
