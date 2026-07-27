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

export function Toolbar() {
  const { state, dispatch } = useAppContext();
  const { schedule, currentView, showFlightStints } = state;
  const has = !!schedule;
  const { runCheck, isChecking } = useBackendConstraintCheck();

  const mkBtn = (bg: string): React.CSSProperties => ({
    padding: '4px 10px',
    backgroundColor: bg,
    color: '#fff',
    border: 'none',
    borderRadius: 3,
    cursor: has ? 'pointer' : 'default',
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
    opacity: has ? 1 : 0.4,
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
          + 割付追加
        </button>
        <button style={mkBtn(palette.accentDark)} disabled={!has}
          onClick={() => dispatch({ type: 'OPEN_NEW_SCHEDULE_DIALOG' })}>
          + 新規製番追加
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
          {isChecking ? '⏳ チェック中...' : '☑ 制約チェック'}
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
          ✈ 出入国バー
        </button>
        <div style={{ marginLeft: 'auto' }}>
          <button
            disabled={!has}
            onClick={() => dispatch({ type: 'OPEN_SEND_TO_SCHEDULER_DIALOG' })}
            style={S.submitBtn(has)}
          >
            ▶ 計画管理ツールへ送信
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