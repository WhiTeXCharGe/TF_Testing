import { forwardRef, useEffect, useMemo, useRef, useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { generateDateRange } from '../../utils/dateUtils';

const panelStyle: React.CSSProperties = {
  width: 300,
  position: 'absolute',
  right: 0,
  top: 0,
  bottom: 0,
  zIndex: 20,
  borderLeft: '1px solid #ccc',
  backgroundColor: '#fafafa',
  padding: 12,
  overflowY: 'auto',
  fontFamily: 'MS Gothic, monospace',
  fontSize: 12,
};

const titleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 'bold',
  marginBottom: 12,
  color: '#1565c0',
  borderBottom: '1px solid #ccc',
  paddingBottom: 4,
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  marginBottom: 10,
};

const labelStyle: React.CSSProperties = { color: '#666', marginBottom: 2 };
const valueStyle: React.CSSProperties = { color: '#222', fontWeight: 'bold' };

const inputStyle: React.CSSProperties = {
  padding: '4px 6px',
  border: '1px solid #b8c6d5',
  borderRadius: 3,
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
};

const deleteBtn: React.CSSProperties = {
  width: '100%',
  padding: '6px 12px',
  backgroundColor: '#c62828',
  color: '#fff',
  border: 'none',
  borderRadius: 3,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
  marginTop: 16,
};

export function SidePanel() {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, selectedAssignmentIndex, selectedUnavailableInfo, violations } = state;
  const panelRef = useRef<HTMLDivElement>(null);

  const isOpen = selectedAssignmentIndex !== null || selectedUnavailableInfo !== null;

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: MouseEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) {
        dispatch({ type: 'SELECT_ASSIGNMENT', payload: null });
        dispatch({ type: 'SELECT_UNAVAILABLE', payload: null });
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [isOpen, dispatch]);

  const assignment = selectedAssignmentIndex !== null && schedule
    ? schedule.assignmentList[selectedAssignmentIndex]
    : null;

  const taskInfo = useMemo(() => {
    if (!schedule || !assignment) return null;

    for (const wt of schedule.workflowTaskList) {
      if (wt.phaseTaskList.length === 0) {
        if (wt.id === assignment.operationTask) {
          return { type: 'misc' as const, workflowTask: wt };
        }
        continue;
      }
      for (const pt of wt.phaseTaskList) {
        for (const ot of pt.operationTaskList) {
          if (ot.id === assignment.operationTask) {
            return { type: 'work' as const, workflowTask: wt, phaseTask: pt, operationTask: ot };
          }
        }
      }
    }
    return null;
  }, [schedule, assignment]);

  if (!isOpen || !envConfig) return null;

  if (selectedUnavailableInfo) {
    return <UnavailablePanel ref={panelRef} />;
  }

  if (!assignment || selectedAssignmentIndex === null) return null;

  if (taskInfo?.type === 'misc') {
    return <MiscPanel ref={panelRef} assignmentIndex={selectedAssignmentIndex} />;
  }

  return <WorkTaskPanel ref={panelRef} assignmentIndex={selectedAssignmentIndex} />;
}

// ── Work Task Panel ──────────────────────────────────────────────────────────

const WorkTaskPanel = forwardRef<HTMLDivElement, { assignmentIndex: number }>(
  function WorkTaskPanel({ assignmentIndex }, ref) {
    const { state, dispatch } = useAppContext();
    const { schedule, envConfig, violations } = state;
    const assignment = schedule?.assignmentList[assignmentIndex];

    const [colorDraft, setColorDraft] = useState('');
    const [descDraft, setDescDraft] = useState(assignment?.description ?? '');

    const taskInfo = useMemo(() => {
      if (!schedule || !assignment) return null;
      for (const wt of schedule.workflowTaskList) {
        for (const pt of wt.phaseTaskList) {
          for (const ot of pt.operationTaskList) {
            if (ot.id === assignment.operationTask) {
              const fab = envConfig?.fabList.find(f => f.id === wt.fab);
              const region = fab ? envConfig?.regionList.find(r => r.id === fab.region) : undefined;
              return { wt, pt, ot, fabName: fab?.name ?? wt.fab ?? '', regionName: region?.name ?? fab?.region ?? '' };
            }
          }
        }
      }
      return null;
    }, [schedule, assignment, envConfig]);

    useEffect(() => {
      if (taskInfo?.ot) {
        setColorDraft(taskInfo.ot.colorCode ?? 'FFFFFF');
      }
    }, [taskInfo?.ot?.id]);

    useEffect(() => {
      setDescDraft(assignment?.description ?? '');
    }, [assignmentIndex]);

    if (!assignment || !schedule) return null;

    const worker = envConfig?.workerList.find(w => w.id === assignment.worker);
    const workerCompany = envConfig?.workerCompanyList.find(c => c.id === worker?.workerCompany);
    const assignmentViolations = violations.filter(v => v.assignmentIndices.includes(assignmentIndex));

    const handleColorSave = () => {
      if (taskInfo?.ot) {
        dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: taskInfo.ot.id, colorCode: colorDraft } });
      }
    };

    const handleDelete = () => {
      if (window.confirm(UI.deleteConfirm)) {
        dispatch({ type: 'DELETE_ASSIGNMENT', payload: assignmentIndex });
      }
    };

    return (
      <div ref={ref} style={panelStyle} onClick={e => e.stopPropagation()}>
        <div style={titleStyle}>作業情報</div>

        <div style={rowStyle}>
          <span style={labelStyle}>作業者</span>
          <span style={valueStyle}>{worker?.name ?? assignment.worker}</span>
          <span style={{ color: '#888' }}>{workerCompany?.name ?? worker?.workerCompany ?? ''}</span>
        </div>

        {taskInfo && (
          <>
            <div style={rowStyle}>
              <span style={labelStyle}>製番</span>
              <span style={valueStyle}>{taskInfo.wt.name ?? taskInfo.wt.id}</span>
            </div>

            <div style={rowStyle}>
              <span style={labelStyle}>FAB / Region</span>
              <span style={valueStyle}>{taskInfo.fabName}</span>
              <span style={{ color: '#888' }}>{taskInfo.regionName}</span>
            </div>

            <div style={rowStyle}>
              <span style={labelStyle}>工程 / 作業</span>
              <span style={valueStyle}>{taskInfo.pt.name ?? taskInfo.pt.id}</span>
              <span style={{ color: '#888' }}>{taskInfo.ot.name ?? taskInfo.ot.id}</span>
            </div>
          </>
        )}

        <div style={rowStyle}>
          <span style={labelStyle}>期間</span>
          <span style={valueStyle}>{assignment.startDate} 〜 {assignment.endDate}</span>
        </div>

        <WorkHourTable assignment={assignment} assignmentIndex={assignmentIndex} />

        <div style={rowStyle}>
          <span style={labelStyle}>バーカラー</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              type="color"
              value={`#${colorDraft.padEnd(6, '0')}`}
              onChange={e => {
                const hex = e.target.value.slice(1).toUpperCase();
                setColorDraft(hex);
                if (taskInfo?.ot) {
                  dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: taskInfo.ot.id, colorCode: hex } });
                }
              }}
              style={{ width: 44, height: 32, padding: 2, border: '1px solid #ccc', borderRadius: 4, cursor: 'pointer', background: 'none' }}
              title="クリックしてカラーを選択"
            />
            <span style={{ color: '#666', fontSize: 11 }}>#{colorDraft}</span>
          </div>
        </div>

        {assignmentViolations.length > 0 && (
          <div style={{ marginTop: 8, backgroundColor: '#fff3e0', border: '1px solid #f57c00', borderRadius: 3, padding: 6 }}>
            <div style={{ color: '#e65100', fontWeight: 'bold', marginBottom: 4 }}>{UI.violationTitle}</div>
            {assignmentViolations.map((v, i) => (
              <div key={i} style={{ fontSize: 11, color: '#c62828', marginBottom: 3 }}>
                <div>{v.message}</div>
                {v.date && <div style={{ color: '#8d1f1f' }}>{UI.violationTargetDateLabel}: {v.date}</div>}
              </div>
            ))}
          </div>
        )}

        <div style={rowStyle}>
          <span style={labelStyle}>備考</span>
          <textarea
            value={descDraft}
            onChange={e => setDescDraft(e.target.value)}
            onBlur={() => dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: assignmentIndex, updates: { description: descDraft } } })}
            rows={3}
            style={{
              ...inputStyle,
              width: '100%',
              resize: 'vertical',
              fontFamily: 'Meiryo, sans-serif',
              fontSize: 12,
            }}
            placeholder="備考を入力..."
          />
        </div>

        <button onClick={handleDelete} style={deleteBtn}>{UI.deleteButton}</button>
      </div>
    );
  },
);

// ── Work Hour Table ───────────────────────────────────────────────────────────

function WorkHourTable({
  assignment,
  assignmentIndex,
}: {
  assignment: { startDate: string; endDate: string; workDateList: Array<{ date: string; hour: number }> };
  assignmentIndex: number;
}) {
  const { dispatch } = useAppContext();

  const dates = useMemo(
    () => generateDateRange(assignment.startDate, assignment.endDate),
    [assignment.startDate, assignment.endDate],
  );

  const hourMap = useMemo(() => {
    const m = new Map<string, number>();
    for (const wd of assignment.workDateList) {
      m.set(wd.date.replace(/\//g, '-'), wd.hour);
    }
    return m;
  }, [assignment.workDateList]);

  const commitHour = (date: string, newHour: number) => {
    const clamped = Math.max(0, Math.min(24, isNaN(newHour) ? 0 : newHour));
    const existing = hourMap.get(date) ?? 0;
    if (clamped === existing) return;
    const newList = dates
      .map(d => {
        const h = d === date ? clamped : (hourMap.get(d) ?? 0);
        return h > 0 ? { date: d, hour: h } : null;
      })
      .filter((x): x is { date: string; hour: number } => x !== null);
    dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: assignmentIndex, updates: { workDateList: newList } } });
  };

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ color: '#666', fontSize: 11, marginBottom: 4 }}>稼働時間 / 日</div>
      <div style={{
        maxHeight: 160,
        overflowY: 'auto',
        border: '1px solid #dde5ef',
        borderRadius: 3,
        fontSize: 11,
        fontFamily: 'Meiryo, sans-serif',
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 72px',
          backgroundColor: '#e8eef6',
          padding: '3px 8px',
          fontWeight: 'bold',
          color: '#1c2b3a',
          position: 'sticky',
          top: 0,
        }}>
          <span>日付</span>
          <span style={{ textAlign: 'right' }}>時間</span>
        </div>
        {dates.map(date => {
          const hour = hourMap.get(date) ?? 0;
          const [, mm, dd] = date.split('-');
          const isZero = hour === 0;
          return (
            <div
              key={date}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 72px',
                padding: '1px 8px',
                backgroundColor: isZero ? '#fdf5f5' : 'transparent',
                borderTop: '1px solid #edf2f8',
                alignItems: 'center',
              }}
            >
              <span style={{ color: isZero ? '#b0a0a0' : '#1c2b3a' }}>{mm}/{dd}</span>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 2 }}>
                <input
                  type="number"
                  min={0}
                  max={24}
                  defaultValue={hour}
                  key={`${date}-${hour}`}
                  onBlur={e => commitHour(date, Number(e.target.value))}
                  onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                  style={{
                    width: 40,
                    padding: '1px 4px',
                    border: '1px solid #c8d5e5',
                    borderRadius: 3,
                    fontSize: 11,
                    fontFamily: 'Meiryo, sans-serif',
                    textAlign: 'right',
                    color: isZero ? '#b0a0a0' : '#1c2b3a',
                    backgroundColor: isZero ? '#fdf5f5' : '#fff',
                    outline: 'none',
                  }}
                />
                <span style={{ color: '#888', fontSize: 10, width: 10 }}>h</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Misc Task Panel ───────────────────────────────────────────────────────────

const MiscPanel = forwardRef<HTMLDivElement, { assignmentIndex: number }>(
  function MiscPanel({ assignmentIndex }, ref) {
    const { state, dispatch } = useAppContext();
    const { schedule, envConfig } = state;
    const assignment = schedule?.assignmentList[assignmentIndex];

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [colorDraft, setColorDraft] = useState('');

    const miscTask = schedule?.workflowTaskList.find(wt => wt.id === assignment?.operationTask);

    useEffect(() => {
      if (assignment) {
        setStartDate(assignment.startDate);
        setEndDate(assignment.endDate);
      }
    }, [assignment?.startDate, assignment?.endDate]);

    useEffect(() => {
      setColorDraft(miscTask?.colorCode ?? 'AAAAAA');
    }, [miscTask?.id]);


    if (!assignment || !schedule) return null;

    const worker = envConfig?.workerList.find(w => w.id === assignment.worker);
    const workerCompany = envConfig?.workerCompanyList.find(c => c.id === worker?.workerCompany);

    const commitDates = (start: string, end: string) => {
      if (!start || !end || start > end) return;
      dispatch({
        type: 'UPDATE_ASSIGNMENT',
        payload: { index: assignmentIndex, updates: { startDate: start, endDate: end } },
      });
    };

    const handleDelete = () => {
      if (window.confirm(UI.deleteConfirm)) {
        dispatch({ type: 'DELETE_ASSIGNMENT', payload: assignmentIndex });
      }
    };

    return (
      <div ref={ref} style={panelStyle} onClick={e => e.stopPropagation()}>
        <div style={titleStyle}>その他作業情報</div>

        <div style={rowStyle}>
          <span style={labelStyle}>作業者</span>
          <span style={valueStyle}>{worker?.name ?? assignment.worker}</span>
          <span style={{ color: '#888' }}>{workerCompany?.name ?? worker?.workerCompany ?? ''}</span>
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>タスク</span>
          <span style={valueStyle}>{miscTask?.name ?? assignment.operationTask}</span>
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>開始日</span>
          <input
            type="date"
            style={inputStyle}
            value={startDate}
            onChange={e => setStartDate(e.target.value)}
            onBlur={() => commitDates(startDate, endDate)}
          />
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>終了日</span>
          <input
            type="date"
            style={inputStyle}
            value={endDate}
            onChange={e => setEndDate(e.target.value)}
            onBlur={() => commitDates(startDate, endDate)}
          />
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>バーカラー</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              type="color"
              value={`#${colorDraft.padEnd(6, '0')}`}
              onChange={e => {
                const hex = e.target.value.slice(1).toUpperCase();
                setColorDraft(hex);
                if (miscTask) {
                  dispatch({ type: 'UPDATE_WORKFLOW_TASK_COLOR', payload: { workflowTaskId: miscTask.id, colorCode: hex } });
                }
              }}
              style={{ width: 44, height: 32, padding: 2, border: '1px solid #ccc', borderRadius: 4, cursor: 'pointer', background: 'none' }}
              title="クリックしてカラーを選択"
            />
            <span style={{ color: '#666', fontSize: 11 }}>#{colorDraft}</span>
          </div>
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>備考</span>
          <textarea
            key={assignmentIndex}
            defaultValue={assignment.description ?? ''}
            onBlur={e => dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: assignmentIndex, updates: { description: e.target.value } } })}
            rows={3}
            style={{
              ...inputStyle,
              width: '100%',
              resize: 'vertical',
              fontFamily: 'Meiryo, sans-serif',
              fontSize: 12,
            }}
            placeholder="備考を入力..."
          />
        </div>

        <button onClick={handleDelete} style={deleteBtn}>{UI.deleteButton}</button>
      </div>
    );
  },
);

// ── Unavailable Date Panel ───────────────────────────────────────────────────

const UnavailablePanel = forwardRef<HTMLDivElement>(
  function UnavailablePanel(_, ref) {
    const { state, dispatch } = useAppContext();
    const { selectedUnavailableInfo, envConfig } = state;
    const [startDraft, setStartDraft] = useState('');
    const [endDraft, setEndDraft] = useState('');

    useEffect(() => {
      if (selectedUnavailableInfo) {
        setStartDraft(selectedUnavailableInfo.startDate);
        setEndDraft(selectedUnavailableInfo.endDate);
      }
    }, [selectedUnavailableInfo?.startDate, selectedUnavailableInfo?.endDate]);

    if (!selectedUnavailableInfo) return null;

    const { workerId, startDate: origStart, endDate: origEnd } = selectedUnavailableInfo;
    const worker = envConfig?.workerList.find(w => w.id === workerId);
    const workerCompany = envConfig?.workerCompanyList.find(c => c.id === worker?.workerCompany);

    const commitRange = (s: string, e: string) => {
      if (!s || !e || s > e) return;
      if (s === origStart && e === origEnd) return;
      dispatch({ type: 'RESIZE_UNAVAILABLE_RANGE', payload: { workerId, oldStartDate: origStart, oldEndDate: origEnd, newStartDate: s, newEndDate: e } });
    };

    const handleDelete = () => {
      if (window.confirm('この休日を削除しますか？')) {
        dispatch({ type: 'DELETE_UNAVAILABLE_RANGE', payload: { workerId, startDate: origStart, endDate: origEnd } });
      }
    };

    return (
      <div ref={ref} style={panelStyle} onClick={e => e.stopPropagation()}>
        <div style={titleStyle}>休日情報</div>

        <div style={rowStyle}>
          <span style={labelStyle}>作業者</span>
          <span style={valueStyle}>{worker?.name ?? workerId}</span>
          <span style={{ color: '#888' }}>{workerCompany?.name ?? worker?.workerCompany ?? ''}</span>
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>開始日</span>
          <input
            type="date"
            style={inputStyle}
            value={startDraft}
            onChange={e => setStartDraft(e.target.value)}
            onBlur={() => commitRange(startDraft, endDraft)}
          />
        </div>

        <div style={rowStyle}>
          <span style={labelStyle}>終了日</span>
          <input
            type="date"
            style={inputStyle}
            value={endDraft}
            onChange={e => setEndDraft(e.target.value)}
            onBlur={() => commitRange(startDraft, endDraft)}
          />
        </div>

        <button onClick={handleDelete} style={deleteBtn}>削除</button>
      </div>
    );
  },
);
