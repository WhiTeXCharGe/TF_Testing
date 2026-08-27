import { useAppContext } from '../../context/AppContext';
import { ViewButtons } from './ViewButtons';
import { UndoRedoButtons } from './UndoRedoButtons';
import { PlanFlexBulkSettings } from './PlanFlexBulkSettings';
import { PlanRangeEditDialog } from './PlanRangeEditDialog';
import { WorkerViewFilter } from './WorkerViewFilter';
import { ModuleViewFilter } from './ModuleViewFilter';
import { toolbarStyles as S } from '../../styles/toolbar';
import { palette } from '../../styles/common';
import { useBackendConstraintCheck } from '../../hooks/useBackendConstraintCheck';
import { UI } from '../../config/uiText';

export function Toolbar() {
  const { state, dispatch } = useAppContext();
  const { schedule, currentView, showFlightStints } = state;
  const has = !!schedule;
  // 新規製番追加 needs a loaded schedule AND no active session — see the button below.
  const canAddSeiban = has && !state.session;
  const { runCheck, isChecking } = useBackendConstraintCheck();

  const mkBtn = (bg: string, enabled: boolean = has): React.CSSProperties => ({
    padding: '4px 10px',
    backgroundColor: bg,
    color: '#fff',
    border: 'none',
    borderRadius: 3,
    cursor: enabled ? 'pointer' : 'default',
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
    opacity: enabled ? 1 : 0.4,
  });

  return (
    <div style={S.root}>
      {/* Row 1: view toggle + action buttons + submit */}
      <div style={S.row}>
        <ViewButtons />
        <div style={S.divider} />
        <UndoRedoButtons />
        <div style={S.divider} />
        <button style={mkBtn('#2e7d32')} disabled={!has}
          onClick={() => dispatch({ type: 'OPEN_TASK_ADD_DIALOG' })}>
          {UI.addBarBtn}
        </button>
        {/* Session-gated like the File > 開く menu item (which LOAD_FILES is
            also hard-blocked for in AppContext): 新規製番追加's MERGE_DATA does
            sync, so it doesn't diverge participants, but it's a destructive,
            unconfirmed rewrite of the document everyone else is looking at.
            UX affordance only — MERGE_DATA still works normally solo. */}
        <button style={mkBtn(palette.accentDark, canAddSeiban)} disabled={!canAddSeiban}
          onClick={() => dispatch({ type: 'OPEN_NEW_SCHEDULE_DIALOG' })}>
          {UI.addSeibanBtn}
        </button>
        <PlanFlexBulkSettings />
        <PlanRangeEditDialog />
        <div style={S.divider} />
        <button
          disabled={!has || isChecking}
          onClick={runCheck}
          style={{
            padding: '4px 12px',
            backgroundColor: has && !isChecking ? '#1565c0' : '#bdbdbd',
            color: '#fff',
            border: 'none',
            borderRadius: 3,
            cursor: has && !isChecking ? 'pointer' : 'default',
            fontSize: 12,
            fontFamily: 'MS Gothic, monospace',
            opacity: has ? 1 : 0.4,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
          }}
        >
          {isChecking ? UI.checkingLabel : UI.constraintCheckBtn}
        </button>
        <button
          disabled={!has}
          onClick={() => dispatch({ type: 'TOGGLE_FLIGHT_STINTS' })}
          style={{
            padding: '4px 10px',
            backgroundColor: showFlightStints ? '#00796b' : '#78909c',
            color: '#fff',
            border: 'none',
            borderRadius: 3,
            cursor: has ? 'pointer' : 'default',
            fontSize: 12,
            fontFamily: 'MS Gothic, monospace',
            opacity: has ? 1 : 0.4,
          }}
        >
          {UI.flightStintsBtn}
        </button>
        <div style={{ marginLeft: 'auto' }}>
          <button
            disabled={!has}
            onClick={() => dispatch({ type: 'OPEN_SEND_TO_SCHEDULER_DIALOG' })}
            style={S.submitBtn(has)}
          >
            {UI.sendToSchedulerBtn}
          </button>
        </div>
      </div>

      {/* Row 2: view-specific filter bar */}
      {has && (
        currentView === 'worker'
          ? <WorkerViewFilter />
          : <ModuleViewFilter />
      )}
    </div>
  );
}