import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { UI } from '@/config/uiConfig';
import { Icon } from '@/components/common/Icon';
import { useGantt } from '@/hooks/useGantt';
import { useRuns } from '@/hooks/useRuns';
import { getRun } from '@/services/runStore';
import { exportGanttToExcel } from '@/utils/excelExport';
import { toKey, dowAbbr } from '@/utils/dateUtils';
import type { GanttCell, GanttData, Run } from '@/types';

// ─── Gantt table ─────────────────────────────────────────────────────────────

function GanttTable({ data }: { data: GanttData }) {
  const { employees, dates, cells, todayDate } = data;
  const todayKey  = toKey(todayDate);

  type MonthGroup = { label: string; span: number; startIdx: number };
  const monthGroups: MonthGroup[] = [];
  dates.forEach((d, i) => {
    const label = `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}`;
    const last = monthGroups[monthGroups.length - 1];
    if (last && last.label === label) last.span++;
    else monthGroups.push({ label, span: 1, startIdx: i });
  });

  function cellClass(d: Date, cell: GanttCell): string {
    const dk = toKey(d);
    const classes: string[] = ['gantt-cell'];
    if (d.getDay() === 0 || d.getDay() === 6) classes.push('weekend');
    if (cell.type === 'unavailable') classes.push('unavail');
    if (dk === todayKey)  classes.push('today-left');
    return classes.join(' ');
  }

  const dowClass = (d: Date) => d.getDay() === 0 ? 'wd-sun' : d.getDay() === 6 ? 'wd-sat' : '';

  return (
    <table className="gantt-table">
      <thead>
        <tr>
          <th className="col-company" rowSpan={3}>Company</th>
          <th className="col-name"    rowSpan={3}>Name</th>
          <th className="col-role"    rowSpan={3}>Role</th>
          <th className="col-mgr"     rowSpan={3}>Mgr</th>
          {monthGroups.map(g => (
            <th key={g.startIdx} colSpan={g.span}
                style={{ background: 'var(--blue)', color: '#fff', fontSize: 11, height: 22, padding: 0 }}>
              {g.label}
            </th>
          ))}
        </tr>
        <tr>
          {dates.map((d, i) => {
            const isWE = d.getDay() === 0 || d.getDay() === 6;
            return <th key={i} style={{ background: isWE ? 'var(--gantt-weekend)' : undefined }}>{d.getDate()}</th>;
          })}
        </tr>
        <tr>
          {dates.map((d, i) => {
            const isWE = d.getDay() === 0 || d.getDay() === 6;
            return (
              <th key={i} className={dowClass(d)} style={{ background: isWE ? 'var(--gantt-weekend)' : undefined }}>
                {dowAbbr(d)}
              </th>
            );
          })}
        </tr>
      </thead>
      <tbody>
        {employees.map((emp, ri) => (
          <tr key={emp.id}>
            <td className="col-company" style={{ background: emp.companyColor, borderRight: '1px solid var(--bdr)' }}>
              {emp.company}
            </td>
            <td className="col-name">{emp.name}</td>
            <td className="col-role" style={{ color: 'var(--text-sec)' }}>{emp.role}</td>
            <td className="col-mgr">
              {emp.isManager
                ? <span style={{ background: 'var(--blue)', color: '#fff', padding: '1px 5px', borderRadius: 3, fontSize: 10, fontWeight: 700 }}>M</span>
                : null}
            </td>
            {dates.map((d, di) => {
              const cell = cells[ri]?.[di] ?? { type: 'empty' };
              const cls = cellClass(d, cell);
              const bg = cell.type === 'work' && cell.moduleColor ? cell.moduleColor : undefined;
              return (
                <td key={di} className={cls} style={bg ? { background: bg } : undefined}
                    title={cell.moduleCode || undefined}>
                  {cell.type === 'work' && cell.moduleCode
                    ? <span className="cell-code">{cell.moduleCode}</span>
                    : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─── Legend ─────────────────────────────────────────────────────────────────

function GanttLegend() {
  return (
    <div className="gantt-legend">
      <div className="legend-item">
        <div className="legend-swatch" style={{ background: 'var(--gantt-unavail)' }} />
        <span>{UI.gantt.legendUnavailable}</span>
      </div>
      <div className="legend-item">
        <div className="legend-swatch" style={{ background: 'var(--gantt-weekend)', borderColor: '#ccc' }} />
        <span>{UI.gantt.legendWeekend}</span>
      </div>
      <div className="legend-item">
        <div className="legend-line-today" />
        <span>{UI.gantt.legendToday}</span>
      </div>
    </div>
  );
}

// ─── New Run panel (reuse same input files, new plan settings) ───────────────

type PanelStep = 'form' | 'confirm' | 'submitted';

function NewRunPanel({ run }: { run: Run }) {
  const navigate = useNavigate();
  const { addRun } = useRuns();
  const [open, setOpen]   = useState(false);
  const [step, setStep]   = useState<PanelStep>('form');
  const [stage, setStage] = useState<'s1' | 's12'>('s12');
  const [planStart, setPlanStart] = useState('2025-10-01');
  const [planEnd, setPlanEnd]     = useState('2025-11-15');
  const [cutoff, setCutoff]       = useState('2025-10-20');
  const [lockFixed, setLockFixed] = useState(true);
  const [warmStart, setWarmStart] = useState(false);
  const [overtime, setOvertime]   = useState(false);

  function toggle() {
    setOpen(o => !o);
    setStep('form');
  }

  function submit() {
    addRun({
      envName: run.inputEnvName,
      schedName: run.inputSchedName,
      label: stage === 's12' ? 'Stage 1+2 rerun' : 'Stage 1 rerun',
      inputDir: run.inputDir,   // keep the real-YAML link so its Gantt still renders
    });
    setStep('submitted');
  }

  const fmt = (s: string) => s.replace(/-/g, '/');

  return (
    <div className="nr-panel">
      <div className={'nr-panel-hd' + (open ? ' open' : '')} onClick={toggle}>
        <Icon name="plus" size={16} />
        <span className="nr-panel-title">{UI.gantt.newRunTitle}</span>
        <Icon name="chevron-right" size={16} className={open ? 'nr-chev-open' : ''} />
      </div>

      {open && (
        <div className="nr-panel-body">
          {step === 'form' && (
            <>
              <div className="nr-reuse">
                <span className="nr-reuse-label">{UI.gantt.reuseLabel}</span>
                <span className="file-chip"><Icon name="comment" size={11} />{run.inputEnvName}</span>
                <span className="file-chip"><Icon name="comment" size={11} />{run.inputSchedName}</span>
                <span style={{ fontSize: 10, color: 'var(--text-sec)' }}>from {run.folderPath}input/</span>
              </div>

              <div className="nr-grid">
                <div className="nr-fg">
                  <label className="nr-label">{UI.gantt.planStart}</label>
                  <input type="date" className="form-input" value={planStart} onChange={e => setPlanStart(e.target.value)} />
                </div>
                <div className="nr-fg">
                  <label className="nr-label">{UI.gantt.planEnd}</label>
                  <input type="date" className="form-input" value={planEnd} onChange={e => setPlanEnd(e.target.value)} />
                </div>
                <div className="nr-fg">
                  <label className="nr-label">{UI.gantt.cutoff}</label>
                  <input type="date" className="form-input" value={cutoff} onChange={e => setCutoff(e.target.value)} />
                  <span style={{ fontSize: 10, color: 'var(--text-sec)' }}>{UI.gantt.cutoffHint}</span>
                </div>
              </div>

              <label className="nr-label" style={{ display: 'block', marginBottom: 6 }}>{UI.gantt.stageLabel}</label>
              <div className="stage-opts">
                <div className={'stage-opt' + (stage === 's1' ? ' active' : '')} onClick={() => setStage('s1')}>
                  {UI.gantt.stage1} <span style={{ fontSize: 10, opacity: .7 }}>{UI.gantt.stage1Time}</span>
                </div>
                <div className={'stage-opt' + (stage === 's12' ? ' active' : '')} onClick={() => setStage('s12')}>
                  {UI.gantt.stage12} <span style={{ fontSize: 10, opacity: .7 }}>{UI.gantt.stage12Time}</span>
                </div>
              </div>

              <div className="nr-toggles">
                <label><input type="checkbox" checked={lockFixed} onChange={e => setLockFixed(e.target.checked)} />{UI.gantt.optLock}</label>
                <label><input type="checkbox" checked={warmStart} onChange={e => setWarmStart(e.target.checked)} />{UI.gantt.optWarm}</label>
                <label><input type="checkbox" checked={overtime} onChange={e => setOvertime(e.target.checked)} />{UI.gantt.optOvertime}</label>
              </div>

              <div className="nr-bar">
                <button className="btn btn-secondary btn-sm" onClick={toggle}>{UI.gantt.back}</button>
                <button className="btn btn-primary btn-sm" onClick={() => setStep('confirm')}>
                  <Icon name="chevron-right" size={14} />{UI.gantt.review}
                </button>
              </div>
            </>
          )}

          {step === 'confirm' && (
            <>
              <div className="nr-confirm">
                <strong>{UI.gantt.readyToSubmit}</strong><br />
                Input files &nbsp;→&nbsp; {run.folderPath}input/<br />
                Plan range &nbsp;→&nbsp; {fmt(planStart)} – {fmt(planEnd)}<br />
                Cut-off date &nbsp;→&nbsp; <strong>{fmt(cutoff)}</strong><br />
                Solver stage &nbsp;→&nbsp; {stage === 's12' ? `${UI.gantt.stage12} ${UI.gantt.stage12Time}` : `${UI.gantt.stage1} ${UI.gantt.stage1Time}`}<br />
                Lock fixed zone &nbsp;→&nbsp; {lockFixed ? 'Yes' : 'No'}<br />
                Warm start &nbsp;→&nbsp; {warmStart ? 'Yes' : 'No'}<br />
                Allow overtime &nbsp;→&nbsp; {overtime ? 'Yes' : 'No'}
              </div>
              <div className="nr-bar" style={{ marginTop: 12 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => setStep('form')}>{UI.gantt.back}</button>
                <button className="btn btn-primary btn-sm" onClick={submit}>
                  <Icon name="play" size={13} />{UI.gantt.submit}
                </button>
              </div>
            </>
          )}

          {step === 'submitted' && (
            <div className="nr-submitted">
              <Icon name="check" size={15} />&nbsp;<strong>{UI.gantt.submittedTitle}</strong><br />
              {UI.gantt.submittedBody}
              <div style={{ marginTop: 10 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => navigate('/')}>{UI.gantt.backToLog}</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function GanttPage() {
  const { runId = '', view } = useParams<{ runId: string; view: string }>();
  const navigate = useNavigate();
  const run = getRun(runId);
  const { ganttData, loading, error, isMock } = useGantt(run?.inputDir ?? null);

  const viewLabel = view === 'result' ? UI.gantt.viewResult : UI.gantt.viewInput;

  async function handleExport() {
    if (!ganttData) return;
    await exportGanttToExcel(ganttData, `gantt_${runId}_${view ?? 'input'}.xlsx`);
  }

  if (!run) {
    return (
      <div>
        <button className="back-link" onClick={() => navigate('/')}>
          <Icon name="chevron-left" size={14} />{UI.gantt.backLabel}
        </button>
        <p style={{ color: 'var(--red)' }}>Run not found: {runId}</p>
      </div>
    );
  }

  return (
    <div className="gantt-page">
      <button className="back-link" onClick={() => navigate('/')}>
        <Icon name="chevron-left" size={14} />{UI.gantt.backLabel}
      </button>

      <div className="gantt-toolbar">
        <span className="gantt-toolbar-title">
          {run.id} <span style={{ color: 'var(--text-sec)', fontWeight: 400 }}>· {viewLabel}</span>
        </span>
        {isMock && <span className="mock-badge">{UI.gantt.mockNote}</span>}
        <div className="gantt-toolbar-spacer" />
        <button className="btn btn-secondary btn-sm" onClick={handleExport} disabled={!ganttData}>
          <Icon name="download" size={14} />{UI.gantt.downloadExcel}
        </button>
      </div>

      {loading && <p style={{ color: 'var(--text-sec)' }}>Loading schedule data…</p>}
      {error   && <p style={{ color: 'var(--red)' }}>{UI.gantt.loadError}<br /><small>{error}</small></p>}

      {ganttData && (
        <>
          <div className="gantt-scroll-outer">
            <GanttTable data={ganttData} />
          </div>
          <GanttLegend />
          <NewRunPanel run={run} />
        </>
      )}
    </div>
  );
}
