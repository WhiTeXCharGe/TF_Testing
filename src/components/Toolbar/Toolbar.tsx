import { ViewButtons } from './ViewButtons';
import { UndoRedoButtons } from './UndoRedoButtons';
import { SearchBar } from './SearchBar';
import { DisplayPeriodInput } from './DisplayPeriodInput';
import { PlanFlexBulkSettings } from './PlanFlexBulkSettings';
import { useAppContext } from '../../context/AppContext';

function SubmitButton({ has }: { has: boolean }) {
  const { dispatch } = useAppContext();
  return (
    <button
      disabled={!has}
      onClick={() => dispatch({
        type: 'SET_ERROR',
        payload: '計画管理ツールへの送信機能は未実装です（バックエンド接続後に有効になります）',
      })}
      title={has ? '計画管理ツールへ送信（未接続）' : 'ファイルを読み込んでください'}
      style={{
        padding: '4px 14px',
        backgroundColor: has ? '#1565c0' : '#bdbdbd',
        color: '#fff',
        border: 'none',
        borderRadius: 3,
        cursor: has ? 'pointer' : 'default',
        fontSize: 12,
        fontFamily: 'MS Gothic, monospace',
      }}
    >
      ▶ 計画管理ツールへ送信
    </button>
  );
}

export function Toolbar() {
  const { state, dispatch } = useAppContext();

  const has = !!state.schedule;

  const mkBtn = (bg: string): React.CSSProperties => ({
    padding: '4px 10px', backgroundColor: bg, color: '#fff',
    border: 'none', borderRadius: 3,
    cursor: has ? 'pointer' : 'default',
    fontSize: 12, fontFamily: 'MS Gothic, monospace',
    opacity: has ? 1 : 0.4,
  });

  const divider: React.CSSProperties = { width: 1, height: 20, backgroundColor: '#d0d0d0', flexShrink: 0 };

  return (
    <div style={{
      backgroundColor: '#f0f2f5',
      borderBottom: '1px solid #d0d5dd',
      padding: '5px 10px',
      display: 'flex',
      flexDirection: 'column',
      gap: 5,
      flexShrink: 0,
    }}>
      {/* Row 1: search + undo/redo + actions + submit (right) */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <SearchBar />
        <div style={divider} />
        <UndoRedoButtons />
        <div style={divider} />
        {/* 割付追加: assign a worker to an existing operation */}
        <button style={mkBtn('#2e7d32')} disabled={!has}
          onClick={() => dispatch({ type: 'OPEN_TASK_ADD_DIALOG' })}>
          + 割付追加
        </button>
        {/* 新規製番追加: add a new device / workflow task */}
        <button style={mkBtn('#1565c0')} disabled={!has}
          onClick={() => dispatch({ type: 'OPEN_NEW_SCHEDULE_DIALOG' })}>
          + 新規製番追加
        </button>
        <PlanFlexBulkSettings />

        {/* Submit button pushed to far right */}
        <div style={{ marginLeft: 'auto' }}>
          <SubmitButton has={has} />
        </div>
      </div>
      {/* Row 2: view toggle + date range */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <ViewButtons />
        <div style={divider} />
        <DisplayPeriodInput />
      </div>
    </div>
  );
}
