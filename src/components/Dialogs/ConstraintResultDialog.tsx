import { useAppContext } from '../../context/AppContext';
import { Violation } from '../../types/appState';
import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { useBackendConstraintCheck } from '../../hooks/useBackendConstraintCheck';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { UI } from '../../config/uiText';

const VIOLATION_LABEL: Record<string, string> = UI.violationLabels;

interface ViolationRow {
  key: string;
  severity: 'error' | 'warning';
  constraintType: string;
  seiban: string;
  taskName: string;
  workers: string[];
  companies: string[];
  dates: string[];
  assignmentIndices: number[];
}

interface Filters {
  names: string[];
  companies: string[];
  seibans: string[];
  tasks: string[];
  dateFrom: string;
  dateTo: string;
}

const EMPTY_FILTERS: Filters = {
  names: [], companies: [], seibans: [], tasks: [],
  dateFrom: '', dateTo: '',
};

function hasActiveFilters(f: Filters) {
  return f.names.length > 0 || f.companies.length > 0 || f.seibans.length > 0
    || f.tasks.length > 0 || !!f.dateFrom || !!f.dateTo;
}

// ── Multi-select dropdown chip component ────────────────────────────────────

function ChipDropdown({
  label, options, selected, onToggle, onClear,
}: {
  label: string;
  options: string[];
  selected: string[];
  onToggle: (v: string) => void;
  onClear: () => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = selected.length > 0;

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div ref={ref} style={{ position: 'relative', flexShrink: 0 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '3px 8px',
          fontSize: 10,
          border: `1px solid ${active ? '#1565c0' : 'rgba(0,0,0,0.18)'}`,
          borderRadius: 12,
          background: active ? '#e3f0fd' : 'rgba(255,255,255,0.65)',
          color: active ? '#1565c0' : '#546e7a',
          cursor: 'pointer',
          fontFamily: 'Meiryo, sans-serif',
          whiteSpace: 'nowrap',
        }}
      >
        {label}
        {active && (
          <span style={{
            background: '#1565c0', color: '#fff',
            borderRadius: 8, padding: '0 5px', fontSize: 9, fontWeight: 700,
          }}>
            {selected.length}
          </span>
        )}
        <span style={{ fontSize: 8, color: active ? '#1565c0' : '#90a4ae' }}>▼</span>
      </button>

      {open && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, zIndex: 300,
          background: 'rgba(250,252,255,0.97)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(0,0,0,0.12)',
          borderRadius: 6,
          boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
          minWidth: 160, maxWidth: 240,
          maxHeight: 220, overflowY: 'auto',
          marginTop: 3,
        }}>
          {options.length === 0 ? (
            <div style={{ padding: '8px 12px', fontSize: 11, color: '#90a4ae' }}>{UI.noOptionsLabel}</div>
          ) : (
            <>
              {active && (
                <div
                  onClick={() => { onClear(); setOpen(false); }}
                  style={{
                    padding: '5px 12px', fontSize: 10, color: '#c62828',
                    cursor: 'pointer', borderBottom: '1px solid #f0f0f0',
                  }}
                >
                  {UI.chipClearAll}
                </div>
              )}
              {options.map(opt => (
                <label
                  key={opt}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '5px 12px', fontSize: 11, cursor: 'pointer',
                    background: selected.includes(opt) ? 'rgba(21,101,192,0.08)' : 'transparent',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(opt)}
                    onChange={() => onToggle(opt)}
                    style={{ margin: 0 }}
                  />
                  <span style={{
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    maxWidth: 180, color: '#37474f',
                  }} title={opt}>{opt}</span>
                </label>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Filter bar ───────────────────────────────────────────────────────────────

function FilterBar({
  filters, options, onChange, onClearAll,
}: {
  filters: Filters;
  options: { names: string[]; companies: string[]; seibans: string[]; tasks: string[] };
  onChange: (patch: Partial<Filters>) => void;
  onClearAll: () => void;
}) {
  const toggle = (key: keyof Pick<Filters, 'names' | 'companies' | 'seibans' | 'tasks'>, v: string) => {
    const cur = filters[key];
    onChange({ [key]: cur.includes(v) ? cur.filter(x => x !== v) : [...cur, v] });
  };

  const active = hasActiveFilters(filters);

  return (
    <div style={{
      padding: '7px 10px',
      borderBottom: '1px solid rgba(255,255,255,0.35)',
      background: 'rgba(248,250,253,0.5)',
    }}>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center' }}>
        <ChipDropdown
          label={UI.dialogWorkerLabel}
          options={options.names}
          selected={filters.names}
          onToggle={v => toggle('names', v)}
          onClear={() => onChange({ names: [] })}
        />
        <ChipDropdown
          label={UI.companyLabel}
          options={options.companies}
          selected={filters.companies}
          onToggle={v => toggle('companies', v)}
          onClear={() => onChange({ companies: [] })}
        />
        <ChipDropdown
          label={UI.deviceCodeLabel}
          options={options.seibans}
          selected={filters.seibans}
          onToggle={v => toggle('seibans', v)}
          onClear={() => onChange({ seibans: [] })}
        />
        <ChipDropdown
          label={UI.taskLabel}
          options={options.tasks}
          selected={filters.tasks}
          onToggle={v => toggle('tasks', v)}
          onClear={() => onChange({ tasks: [] })}
        />
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 5, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 10, color: '#90a4ae', flexShrink: 0 }}>{UI.dateRangeLabel}</span>
        <input
          type="date"
          value={filters.dateFrom}
          onChange={e => onChange({ dateFrom: e.target.value })}
          style={dateInputStyle}
        />
        <span style={{ fontSize: 10, color: '#90a4ae' }}>{UI.periodSeparator}</span>
        <input
          type="date"
          value={filters.dateTo}
          onChange={e => onChange({ dateTo: e.target.value })}
          style={dateInputStyle}
        />
        {active && (
          <button
            onClick={onClearAll}
            style={{
              marginLeft: 4,
              fontSize: 10, padding: '3px 10px',
              border: '1px solid rgba(0,0,0,0.15)',
              background: 'rgba(255,255,255,0.7)',
              borderRadius: 12, cursor: 'pointer', color: '#c62828',
              fontFamily: 'Meiryo, sans-serif',
            }}
          >
            {UI.clearAllFiltersBtn}
          </button>
        )}
      </div>
    </div>
  );
}

const dateInputStyle: React.CSSProperties = {
  padding: '2px 6px',
  fontSize: 10,
  border: '1px solid rgba(0,0,0,0.15)',
  borderRadius: 4,
  background: 'rgba(255,255,255,0.65)',
  fontFamily: 'Meiryo, sans-serif',
  color: '#37474f',
  outline: 'none',
};

// ── Violation grouping ───────────────────────────────────────────────────────

function buildLookups(schedule: ScheduleData, envConfig: EnvConfig) {
  const opTaskInfo = new Map<string, { seiban: string; taskName: string }>();
  for (const wt of schedule.workflowTaskList) {
    const seiban = wt.name ?? wt.id;
    if (wt.phaseTaskList.length === 0) {
      opTaskInfo.set(wt.id, { seiban, taskName: wt.name ?? wt.id });
    } else {
      for (const pt of wt.phaseTaskList) {
        for (const ot of pt.operationTaskList) {
          opTaskInfo.set(ot.id, { seiban, taskName: ot.name ?? ot.operation ?? ot.id });
        }
      }
    }
  }
  const workerInfo = new Map(
    envConfig.workerList.map(w => {
      const company = envConfig.workerCompanyList.find(c => c.id === w.workerCompany);
      return [w.id, { name: w.name ?? w.id, company: company?.name ?? w.workerCompany ?? '' }];
    }),
  );
  const assignmentInfo = schedule.assignmentList.map((a, i) => {
    const task = opTaskInfo.get(a.operationTask) ?? { seiban: a.operationTask, taskName: a.operationTask };
    const worker = workerInfo.get(a.worker) ?? { name: a.worker, company: '' };
    return { index: i, workerId: a.worker, operationTask: a.operationTask, ...task, ...worker };
  });
  return { opTaskInfo, workerInfo, assignmentInfo };
}

function groupViolations(violations: Violation[], schedule: ScheduleData, envConfig: EnvConfig): ViolationRow[] {
  const { assignmentInfo, opTaskInfo, workerInfo } = buildLookups(schedule, envConfig);
  const groups = new Map<string, ViolationRow>();

  for (const v of violations) {
    const type = v.type;
    const indices = v.assignmentIndices;
    let key: string;
    let seiban = '';
    let taskName = '';
    let workers: string[] = [];
    let companies: string[] = [];

    if (type === 'TASK_WORKER_COUNT') {
      const opTask = indices[0] !== undefined ? schedule.assignmentList[indices[0]]?.operationTask : undefined;
      key = `${type}_${opTask ?? 'unknown'}_${v.message.includes('最小') ? 'min' : 'max'}`;
      const info = opTask ? opTaskInfo.get(opTask) : undefined;
      seiban = info?.seiban ?? opTask ?? '';
      taskName = info?.taskName ?? opTask ?? '';
      workers = [...new Set(indices.map(i => assignmentInfo[i]?.name ?? '').filter(Boolean))];
      companies = [...new Set(indices.map(i => assignmentInfo[i]?.company ?? '').filter(Boolean))];
    } else if (type === 'OVERLAP') {
      const sorted = [...indices].sort((a, b) => a - b);
      key = `${type}_${sorted.join('_')}`;
      const infos = sorted.map(i => assignmentInfo[i]).filter(Boolean);
      seiban = infos[0]?.seiban ?? '';
      taskName = infos[0]?.taskName ?? '';
      workers = [...new Set(infos.map(i => i.name))];
      companies = [...new Set(infos.map(i => i.company))];
    } else {
      const idx = indices[0];
      key = `${type}_${idx}`;
      const info = idx !== undefined ? assignmentInfo[idx] : undefined;
      seiban = info?.seiban ?? '';
      taskName = info?.taskName ?? '';
      if (idx !== undefined) {
        const w = workerInfo.get(schedule.assignmentList[idx]?.worker ?? '');
        workers = w ? [w.name] : [];
        companies = w ? [w.company] : [];
      }
    }

    if (groups.has(key)) {
      const g = groups.get(key)!;
      if (v.date && !g.dates.includes(v.date)) g.dates.push(v.date);
      for (const i of indices) {
        if (!g.assignmentIndices.includes(i)) g.assignmentIndices.push(i);
      }
    } else {
      groups.set(key, {
        key,
        severity: v.severity === 'warning' ? 'warning' : 'error',
        constraintType: VIOLATION_LABEL[type] ?? type,
        seiban, taskName, workers, companies,
        dates: v.date ? [v.date] : [],
        assignmentIndices: [...indices],
      });
    }
  }

  return [...groups.values()].sort((a, b) => {
    if (a.severity !== b.severity) return a.severity === 'error' ? -1 : 1;
    return a.seiban.localeCompare(b.seiban, 'ja');
  });
}

// ── Table components ─────────────────────────────────────────────────────────

type ColKey = 'badge' | 'type' | 'seiban' | 'task' | 'worker' | 'date';
const DEFAULT_COL_WIDTHS: Record<ColKey, number> = { badge: 42, type: 76, seiban: 90, task: 90, worker: 90, date: 90 };
const COL_DEFS: Array<{ key: ColKey; label: string; align?: 'center' | 'left' }> = [
  { key: 'badge', label: '', align: 'center' },
  { key: 'type',   label: UI.colConstraintLabel },
  { key: 'seiban', label: UI.deviceCodeLabel },
  { key: 'task',   label: UI.taskLabel },
  { key: 'worker', label: UI.dialogWorkerLabel },
  { key: 'date',   label: UI.dateColumnLabel },
];

function Badge({ severity }: { severity: string }) {
  const isError = severity !== 'warning';
  return (
    <span style={{
      display: 'inline-block', padding: '1px 6px', borderRadius: 8,
      fontSize: 9, fontWeight: 700,
      backgroundColor: isError ? '#fdecea' : '#fff8e1',
      color: isError ? '#c62828' : '#f57f17',
      border: `1px solid ${isError ? '#ef9a9a' : '#ffe082'}`,
      whiteSpace: 'nowrap',
    }}>
      {isError ? UI.badgeError : UI.badgeWarning}
    </span>
  );
}

function ResizableTableHeader({
  colWidths, onResizeCol,
}: {
  colWidths: Record<ColKey, number>;
  onResizeCol: (col: ColKey, w: number) => void;
}) {
  const resizing = useRef<{ col: ColKey; startX: number; startW: number } | null>(null);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!resizing.current) return;
      const { col, startX, startW } = resizing.current;
      onResizeCol(col, Math.max(36, startW + e.clientX - startX));
    };
    const onUp = () => { resizing.current = null; };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [onResizeCol]);

  return (
    <div style={{ display: 'flex', background: 'rgba(236,240,246,0.7)', borderBottom: '1px solid rgba(0,0,0,0.08)', position: 'sticky', top: 0, zIndex: 2, userSelect: 'none' }}>
      {COL_DEFS.map((col, i) => (
        <div key={col.key} style={{ position: 'relative', width: colWidths[col.key], minWidth: colWidths[col.key], flexShrink: 0 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#546e7a', padding: '5px 6px', textAlign: col.align ?? 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {col.label}
          </div>
          {i < COL_DEFS.length - 1 && (
            <div
              onMouseDown={e => {
                resizing.current = { col: col.key, startX: e.clientX, startW: colWidths[col.key] };
                e.preventDefault();
                e.stopPropagation();
              }}
              style={{
                position: 'absolute', right: 0, top: 0, bottom: 0, width: 5,
                cursor: 'col-resize', zIndex: 3,
                background: 'rgba(0,0,0,0)',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(21,101,192,0.25)'; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(0,0,0,0)'; }}
            />
          )}
        </div>
      ))}
    </div>
  );
}

function TableRow({
  row, isSelected, onClick, colWidths,
}: {
  row: ViolationRow; isSelected: boolean; onClick: () => void;
  colWidths: Record<ColKey, number>;
}) {
  const [hovered, setHovered] = useState(false);
  const bg = isSelected ? 'rgba(200,230,201,0.6)' : hovered ? 'rgba(0,0,0,0.03)' : 'transparent';

  const cell = (content: React.ReactNode, col: ColKey, align: 'left' | 'center' = 'left') => (
    <div style={{ width: colWidths[col], minWidth: colWidths[col], fontSize: 11, color: '#37474f', padding: '5px 6px', textAlign: align, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      title={typeof content === 'string' ? content : undefined}>
      {content}
    </div>
  );

  const dateText = row.dates.length === 0 ? '—'
    : row.dates.length <= 3 ? row.dates.join(', ')
    : `${row.dates[0]} ${UI.andNMore(row.dates.length - 1)}`;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ display: 'flex', alignItems: 'center', background: bg, borderBottom: '1px solid rgba(0,0,0,0.05)', cursor: row.assignmentIndices.length > 0 ? 'pointer' : 'default', transition: 'background 0.1s' }}
    >
      <div style={{ width: colWidths.badge, minWidth: colWidths.badge, padding: '5px 4px', display: 'flex', justifyContent: 'center', flexShrink: 0 }}>
        <Badge severity={row.severity} />
      </div>
      {cell(row.constraintType, 'type')}
      {cell(row.seiban, 'seiban')}
      {cell(row.taskName, 'task')}
      {cell(row.workers.join(', '), 'worker')}
      {cell(dateText, 'date')}
    </div>
  );
}

// ── Main dialog ──────────────────────────────────────────────────────────────

export function ConstraintResultDialog() {
  const { state, dispatch } = useAppContext();
  const { isConstraintDialogOpen, isConstraintChecking, backendViolations, constraintCheckedAt, violations, schedule, envConfig } = state;
  const { runCheck } = useBackendConstraintCheck();

  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const patchFilters = (patch: Partial<Filters>) => setFilters(f => ({ ...f, ...patch }));

  const [colWidths, setColWidths] = useState<Record<ColKey, number>>(DEFAULT_COL_WIDTHS);
  const resizeCol = useCallback((col: ColKey, w: number) => setColWidths(cw => ({ ...cw, [col]: w })), []);

  const miscTaskIds = useMemo(() => new Set(
    (schedule?.workflowTaskList ?? []).filter(wt => wt.phaseTaskList.length === 0).map(wt => wt.id),
  ), [schedule]);

  const filteredBackend = useMemo(() => backendViolations.filter(v => {
    if (v.type === 'RESPONSIBLE_WORKER') {
      const isForMisc = v.assignmentIndices.some(idx => {
        const a = schedule?.assignmentList[idx];
        return a != null && miscTaskIds.has(a.operationTask);
      });
      if (isForMisc) return false;
    }
    return true;
  }), [backendViolations, schedule, miscTaskIds]);

  const allViolations = useMemo(() => [...filteredBackend, ...violations], [filteredBackend, violations]);

  const grouped = useMemo(() => {
    if (!schedule || !envConfig) return [];
    return groupViolations(allViolations, schedule, envConfig);
  }, [allViolations, schedule, envConfig]);

  // Derive filter options from all grouped rows (not filtered)
  const filterOptions = useMemo(() => ({
    names:     [...new Set(grouped.flatMap(r => r.workers))].sort((a, b) => a.localeCompare(b, 'ja')),
    companies: [...new Set(grouped.flatMap(r => r.companies))].sort((a, b) => a.localeCompare(b, 'ja')),
    seibans:   [...new Set(grouped.map(r => r.seiban))].filter(Boolean).sort((a, b) => a.localeCompare(b, 'ja')),
    tasks:     [...new Set(grouped.map(r => r.taskName))].filter(Boolean).sort((a, b) => a.localeCompare(b, 'ja')),
  }), [grouped]);

  const filteredRows = useMemo(() => grouped.filter(row => {
    if (filters.names.length > 0 && !row.workers.some(w => filters.names.includes(w))) return false;
    if (filters.companies.length > 0 && !row.companies.some(c => filters.companies.includes(c))) return false;
    if (filters.seibans.length > 0 && !filters.seibans.includes(row.seiban)) return false;
    if (filters.tasks.length > 0 && !filters.tasks.includes(row.taskName)) return false;
    if (filters.dateFrom || filters.dateTo) {
      // Keep row if ANY date is within the range
      const hasDateInRange = row.dates.length === 0
        ? false
        : row.dates.some(d => {
            if (filters.dateFrom && d < filters.dateFrom) return false;
            if (filters.dateTo && d > filters.dateTo) return false;
            return true;
          });
      if (!hasDateInRange) return false;
    }
    return true;
  }), [grouped, filters]);

  const [pos, setPos] = useState({ x: window.innerWidth - 510, y: 60 });
  const [size] = useState({ w: 480, h: 580 });
  const dragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });

  const onMouseDownHeader = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    dragOffset.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
    e.preventDefault();
  }, [pos]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      setPos({
        x: Math.max(0, Math.min(window.innerWidth - size.w, e.clientX - dragOffset.current.x)),
        y: Math.max(0, Math.min(window.innerHeight - 80, e.clientY - dragOffset.current.y)),
      });
    };
    const onUp = () => { dragging.current = false; };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [size.w]);

  if (!isConstraintDialogOpen && !isConstraintChecking) return null;

  const errors   = filteredRows.filter(r => r.severity === 'error');
  const warnings = filteredRows.filter(r => r.severity === 'warning');
  const totalErrors   = grouped.filter(r => r.severity === 'error').length;
  const totalWarnings = grouped.filter(r => r.severity === 'warning').length;

  const handleRowClick = (row: ViolationRow) => {
    if (row.assignmentIndices[0] !== undefined) {
      dispatch({ type: 'SELECT_ASSIGNMENT_AND_SCROLL', payload: row.assignmentIndices[0] });
    }
  };

  return (
    <div
      style={{
        position: 'fixed', left: pos.x, top: pos.y, width: size.w, height: size.h,
        zIndex: 1200, display: 'flex', flexDirection: 'column',
        fontFamily: 'Meiryo, sans-serif',
        background: 'rgba(245,248,252,0.88)',
        backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
        boxShadow: '0 8px 40px rgba(0,0,0,0.18), 0 1px 0 rgba(255,255,255,0.6) inset',
        borderRadius: 10, overflow: 'hidden', resize: 'both',
        minWidth: 360, minHeight: 200, border: 'none',
      }}
      onMouseUp={() => { dragging.current = false; }}
    >
      {/* Header */}
      <div
        onMouseDown={onMouseDownHeader}
        style={{
          padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 10,
          flexShrink: 0, background: 'transparent', cursor: 'move', userSelect: 'none',
          borderBottom: '1px solid rgba(255,255,255,0.35)',
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 700, color: '#1e334b', flex: 1 }}>{UI.constraintDialogTitle}</span>
        {constraintCheckedAt && (
          <span style={{ fontSize: 10, color: '#b0bec5' }}>
            {new Date(constraintCheckedAt).toLocaleTimeString('ja-JP')}
          </span>
        )}
        <button
          onMouseDown={e => e.stopPropagation()}
          onClick={() => dispatch({ type: 'CLOSE_CONSTRAINT_DIALOG' })}
          style={{ border: 'none', background: 'none', cursor: 'pointer', fontSize: 16, color: '#90a4ae', padding: '2px 6px', borderRadius: 4, lineHeight: 1 }}
        >✕</button>
      </div>

      {isConstraintChecking ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, color: '#607d8b' }}>
          <div style={{ fontSize: 28 }}>⏳</div>
          <div style={{ fontSize: 13 }}>{UI.constraintCheckingBody}</div>
        </div>
      ) : (
        <>
          {/* Summary */}
          <div style={{
            padding: '7px 14px',
            background: grouped.length === 0 ? 'rgba(232,245,233,0.5)' : 'transparent',
            borderBottom: '1px solid rgba(255,255,255,0.35)',
            display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, flexWrap: 'wrap',
          }}>
            {grouped.length === 0 ? (
              <span style={{ fontSize: 13, color: '#2e7d32', fontWeight: 700 }}>{UI.noViolations}</span>
            ) : (
              <>
                {totalErrors > 0 && (
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#c62828', background: '#fdecea', padding: '2px 10px', borderRadius: 12, border: '1px solid #ef9a9a' }}>
                    {UI.errorCount(totalErrors)}
                  </span>
                )}
                {totalWarnings > 0 && (
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#f57f17', background: '#fff8e1', padding: '2px 10px', borderRadius: 12, border: '1px solid #ffe082' }}>
                    {UI.warningCount(totalWarnings)}
                  </span>
                )}
                {filteredRows.length !== grouped.length && (
                  <span style={{ fontSize: 10, color: '#90a4ae' }}>{UI.shownCount(filteredRows.length)}</span>
                )}
              </>
            )}
          </div>

          {/* Filters */}
          <FilterBar
            filters={filters}
            options={filterOptions}
            onChange={patchFilters}
            onClearAll={() => setFilters(EMPTY_FILTERS)}
          />

          {/* Table */}
          <div style={{ flex: 1, overflowY: 'auto', overflowX: 'auto' }}>
            {filteredRows.length === 0 ? (
              <div style={{ padding: 40, textAlign: 'center', color: '#90a4ae', fontSize: 13 }}>
                {grouped.length === 0 ? UI.allClearMessage : UI.noMatchingViolations}
              </div>
            ) : (
              <div style={{ minWidth: Object.values(colWidths).reduce((a, b) => a + b, 0) }}>
                <ResizableTableHeader colWidths={colWidths} onResizeCol={resizeCol} />
                {errors.length > 0 && (
                  <>
                    <div style={{ padding: '4px 10px', fontSize: 10, fontWeight: 700, color: '#c62828', background: 'rgba(253,235,234,0.5)', letterSpacing: 1 }}>{UI.errorsSectionLabel}</div>
                    {errors.map(row => (
                      <TableRow key={row.key} row={row} colWidths={colWidths}
                        isSelected={row.assignmentIndices.some(i => i === state.selectedAssignmentIndex)}
                        onClick={() => handleRowClick(row)} />
                    ))}
                  </>
                )}
                {warnings.length > 0 && (
                  <>
                    <div style={{ padding: '4px 10px', fontSize: 10, fontWeight: 700, color: '#f57f17', background: 'rgba(255,248,225,0.5)', letterSpacing: 1 }}>{UI.warningsSectionLabel}</div>
                    {warnings.map(row => (
                      <TableRow key={row.key} row={row} colWidths={colWidths}
                        isSelected={row.assignmentIndices.some(i => i === state.selectedAssignmentIndex)}
                        onClick={() => handleRowClick(row)} />
                    ))}
                  </>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div style={{ padding: '8px 14px', borderTop: '1px solid rgba(0,0,0,0.06)', display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
            {filteredRows.length > 0 && (
              <span style={{ fontSize: 10, color: '#b0bec5', flex: 1 }}>{UI.rowClickHint}</span>
            )}
            <button
              onClick={runCheck}
              disabled={isConstraintChecking}
              style={{
                marginLeft: 'auto', padding: '4px 14px', fontSize: 12,
                fontFamily: 'MS Gothic, monospace',
                background: isConstraintChecking ? '#bdbdbd' : '#1565c0',
                color: '#fff', border: 'none', borderRadius: 4,
                cursor: isConstraintChecking ? 'default' : 'pointer', flexShrink: 0,
              }}
            >
              {isConstraintChecking ? UI.checkingLabel : UI.recheckBtn}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
