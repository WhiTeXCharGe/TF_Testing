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
  const isReadOnly = state.session?.role === 'view';
  const canEdit = has && !isReadOnly;
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
        <button style={mkBtn('#2e7d32')} disabled={!canEdit}
          onClick={() => dispatch({ type: 'OPEN_TASK_ADD_DIALOG' })}>
          {UI.addBarBtn}
        </button>
        {/* Unlike File > 開く (LOAD_FILES, hard-blocked in AppContext because it
            swaps out the whole document from under other participants), both
            of 新規製番追加's actions are additive and already fully synced:
            the form tab's ADD_WORKFLOW_TASKS just appends a new task with a
            fresh id, and the import tab's MERGE_DATA only merges in workflow
            tasks whose id isn't already present and appends new assignments —
            neither ever overwrites existing data. Safe to use during a session. */}
        <button style={mkBtn(palette.accentDark, canEdit)} disabled={!canEdit}
          onClick={() => dispatch({ type: 'OPEN_NEW_SCHEDULE_DIALOG' })}>
          {UI.addSeibanBtn}
        </button>
        {/* PlanFlexBulkSettings/PlanRangeEditDialog render their own trigger
            button internally (no props, no `disabled` to forward), so
            `disabled={!canEdit}` can't be applied to their button the way
            it is above without touching those files. Gating render on
            `canEdit` here — the same condition already used for the row 2
            filter panels below — keeps the fix entirely inside Toolbar.tsx:
            a view-role participant never gets the button in the DOM at all,
            so there's no click *or* keyboard path into BULK_UPDATE_FLEXIBILITY
            / UPDATE_PLAN_RANGE. (`canEdit` already implies `has`, so this
            also preserves today's behavior of hiding both when there's no
            schedule loaded.) */}
        {canEdit && <PlanFlexBulkSettings />}
        {canEdit && <PlanRangeEditDialog />}
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
            disabled={!canEdit}
            onClick={() => dispatch({ type: 'OPEN_SEND_TO_SCHEDULER_DIALOG' })}
            style={S.submitBtn(canEdit)}
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