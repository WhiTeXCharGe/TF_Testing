import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { PlanFlexibility } from '../../types/schedule';

export function SidePanel() {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, selectedAssignmentIndex, violations } = state;

  const panel: React.CSSProperties = {
    width: 240,
    flexShrink: 0,
    borderLeft: '1px solid #ccc',
    backgroundColor: '#fafafa',
    padding: 12,
    overflowY: 'auto',
    fontFamily: 'MS Gothic, monospace',
    fontSize: 12,
  };

  const title: React.CSSProperties = {
    fontSize: 13,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#1565c0',
    borderBottom: '1px solid #ccc',
    paddingBottom: 4,
  };

  const row: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: 10,
  };

  const labelStyle: React.CSSProperties = { color: '#666', marginBottom: 2 };
  const valueStyle: React.CSSProperties = { color: '#222', fontWeight: 'bold' };

  if (selectedAssignmentIndex === null || !schedule) {
    return (
      <div style={panel}>
        <div style={title}>{UI.sidePanelTitle}</div>
        <div style={{ color: '#999', fontSize: 11 }}>{UI.noSelectionMessage}</div>
      </div>
    );
  }

  const assignment = schedule.assignmentList[selectedAssignmentIndex];
  if (!assignment) return null;

  const worker = envConfig?.workerList.find(w => w.id === assignment.worker);
  const workerName = worker?.name ?? assignment.worker;

  // Find operation task name
  let opTaskName = assignment.operationTask;
  let workloadHours = 0;
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      const ot = pt.operationTaskList.find(o => o.id === assignment.operationTask);
      if (ot) { opTaskName = ot.name ?? ot.id; workloadHours = ot.workloadHours; break; }
    }
  }

  const assignmentViolations = violations.filter(v => v.assignmentIndices.includes(selectedAssignmentIndex));

  const handleFlexChange = (value: PlanFlexibility) => {
    dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: selectedAssignmentIndex, updates: { planFlexibility: value } } });
  };

  const handleDelete = () => {
    if (window.confirm('選択したタスクを削除しますか？')) {
      dispatch({ type: 'DELETE_ASSIGNMENT', payload: selectedAssignmentIndex });
    }
  };

  return (
    <div style={panel}>
      <div style={title}>{UI.sidePanelTitle}</div>

      <div style={row}>
        <span style={labelStyle}>{UI.workerLabel}</span>
        <span style={valueStyle}>{workerName}</span>
      </div>
      <div style={row}>
        <span style={labelStyle}>{UI.taskLabel}</span>
        <span style={valueStyle}>{opTaskName}</span>
      </div>
      <div style={row}>
        <span style={labelStyle}>{UI.startLabel}</span>
        <span style={valueStyle}>{assignment.startDate}</span>
      </div>
      <div style={row}>
        <span style={labelStyle}>{UI.endLabel}</span>
        <span style={valueStyle}>{assignment.endDate}</span>
      </div>
      <div style={row}>
        <span style={labelStyle}>{UI.workloadLabel}</span>
        <span style={valueStyle}>{workloadHours} {UI.hoursUnit}</span>
      </div>
      <div style={row}>
        <span style={labelStyle}>{UI.flexibilityLabel}</span>
        <select
          value={assignment.planFlexibility}
          onChange={e => handleFlexChange(e.target.value as PlanFlexibility)}
          style={{ padding: '3px', fontFamily: 'MS Gothic, monospace', fontSize: 12 }}
        >
          <option value="Flexible">{UI.flexible}</option>
          <option value="Reluctant">{UI.reluctant}</option>
          <option value="Fixed">{UI.fixed}</option>
        </select>
      </div>

      {assignmentViolations.length > 0 && (
        <div style={{ marginTop: 8, backgroundColor: '#fff3e0', border: '1px solid #f57c00', borderRadius: 3, padding: 6 }}>
          <div style={{ color: '#e65100', fontWeight: 'bold', marginBottom: 4 }}>⚠ 制約違反</div>
          {assignmentViolations.map((v, i) => (
            <div key={i} style={{ fontSize: 11, color: '#c62828', marginBottom: 2 }}>{v.message}</div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        <button
          onClick={handleDelete}
          style={{
            padding: '4px 12px',
            backgroundColor: '#c62828',
            color: '#fff',
            border: 'none',
            borderRadius: 3,
            cursor: 'pointer',
            fontSize: 12,
            fontFamily: 'MS Gothic, monospace',
            width: '100%',
          }}
        >
          {UI.deleteButton}
        </button>
      </div>
    </div>
  );
}
