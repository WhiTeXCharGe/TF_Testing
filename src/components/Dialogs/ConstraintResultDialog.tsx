import { useAppContext } from '../../context/AppContext';
import { Violation } from '../../types/appState';
import { useRef, useState, useCallback, useEffect } from 'react';
import { useBackendConstraintCheck } from '../../hooks/useBackendConstraintCheck';

const VIOLATION_LABEL: Record<string, string> = {
  OVERLAP: '同一日重複',
  WORKER_UNAVAILABLE: '作業不可日',
  PHASE_OVERRUN: '工程日付超過',
  WORK_HOUR_RANGE: '作業時間範囲',
  SKILL_MISMATCH: 'スキル不足',
  TASK_WORKER_COUNT: '作業者数',
  PHASE_SEQUENCE: '工程順序',
  WORKLOAD_TOTAL: '必要作業量',
  RESPONSIBLE_WORKER: '作業責任者',
  TRAVEL_DAYS: '移動日',
};

function Badge({ severity }: { severity?: string }) {
  const isError = severity !== 'warning';
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 7px',
      borderRadius: 10,
      fontSize: 10,
      fontWeight: 700,
      backgroundColor: isError ? '#fdecea' : '#fff8e1',
      color: isError ? '#c62828' : '#f57f17',
      border: `1px solid ${isError ? '#ef9a9a' : '#ffe082'}`,
      flexShrink: 0,
    }}>
      {isError ? 'ERROR' : 'WARN'}
    </span>
  );
}

function ViolationRow({ v, onClick, isSelected }: { v: Violation; onClick: () => void; isSelected: boolean }) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: '8px 12px',
        borderBottom: '1px solid #f0f4f8',
        cursor: v.assignmentIndices.length > 0 ? 'pointer' : 'default',
        background: isSelected ? '#e8f5e9' : 'transparent',
        transition: 'background 0.1s',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Badge severity={v.severity} />
        <span style={{
          fontSize: 10,
          color: '#607d8b',
          background: '#eceff1',
          borderRadius: 4,
          padding: '1px 6px',
          fontFamily: 'monospace',
          flexShrink: 0,
        }}>
          {VIOLATION_LABEL[v.type] ?? v.type}
        </span>
        {v.date && (
          <span style={{ fontSize: 10, color: '#90a4ae', marginLeft: 'auto' }}>{v.date}</span>
        )}
      </div>
      <div style={{ fontSize: 12, color: '#37474f', lineHeight: 1.5 }}>{v.message}</div>
    </div>
  );
}

export function ConstraintResultDialog() {
  const { state, dispatch } = useAppContext();
  const { isConstraintDialogOpen, isConstraintChecking, backendViolations, constraintCheckedAt, violations, schedule } = state;
  const { runCheck } = useBackendConstraintCheck();

  // Misc task IDs — no manager/workload constraints apply to them
  const miscTaskIds = new Set(
    (schedule?.workflowTaskList ?? [])
      .filter(wt => wt.phaseTaskList.length === 0)
      .map(wt => wt.id),
  );

  // Filter backend violations:
  // - WORKLOAD_TOTAL: replaced by correct frontend calculation
  // - RESPONSIBLE_WORKER on misc tasks: misc tasks don't require a manager
  const filteredBackend = backendViolations.filter(v => {
    if (v.type === 'WORKLOAD_TOTAL') return false;
    if (v.type === 'RESPONSIBLE_WORKER') {
      const isForMisc = v.assignmentIndices.some(idx => {
        const a = schedule?.assignmentList[idx];
        return a != null && miscTaskIds.has(a.operationTask);
      });
      if (isForMisc) return false;
    }
    return true;
  });

  // Combine filtered backend violations and frontend violations
  const allViolations = [...filteredBackend, ...violations];

  const [pos, setPos] = useState({ x: window.innerWidth - 460, y: 60 });
  const [size, setSize] = useState({ w: 420, h: 560 });
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

  const errors = allViolations.filter(v => v.severity !== 'warning');
  const warnings = allViolations.filter(v => v.severity === 'warning');

  const handleViolationClick = (v: Violation) => {
    if (v.assignmentIndices.length > 0 && v.assignmentIndices[0] !== undefined) {
      dispatch({ type: 'SELECT_ASSIGNMENT', payload: v.assignmentIndices[0] });
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        left: pos.x,
        top: pos.y,
        width: size.w,
        height: size.h,
        zIndex: 1200,
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'Meiryo, sans-serif',
        background: 'rgba(245,248,252,0.82)',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        boxShadow: '0 8px 40px rgba(0,0,0,0.18), 0 1px 0 rgba(255,255,255,0.6) inset',
        borderRadius: 10,
        overflow: 'hidden',
        resize: 'both',
        minWidth: 300,
        minHeight: 200,
        border: 'none',
      }}
      onMouseUp={() => { dragging.current = false; }}
    >
      {/* Draggable header */}
      <div
        onMouseDown={onMouseDownHeader}
        style={{
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexShrink: 0,
          background: 'transparent',
          cursor: 'move',
          userSelect: 'none',
          borderBottom: '1px solid rgba(255,255,255,0.35)',
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 700, color: '#1e334b', flex: 1 }}>
          制約チェック結果
        </span>
        <button
          onMouseDown={e => e.stopPropagation()}
          onClick={() => dispatch({ type: 'CLOSE_CONSTRAINT_DIALOG' })}
          style={{
            border: 'none', background: 'none', cursor: 'pointer',
            fontSize: 16, color: '#90a4ae', padding: '2px 6px', borderRadius: 4,
            lineHeight: 1,
          }}
        >
          ✕
        </button>
      </div>

      {/* Loading state */}
      {isConstraintChecking ? (
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 16, color: '#607d8b',
        }}>
          <div style={{ fontSize: 28 }}>⏳</div>
          <div style={{ fontSize: 13 }}>バックエンドで制約チェック中...</div>
        </div>
      ) : (
        <>
          {/* Summary bar */}
          <div style={{
            padding: '8px 14px',
            background: allViolations.length === 0 ? 'rgba(232,245,233,0.5)' : 'transparent',
            borderBottom: '1px solid rgba(255,255,255,0.35)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            flexShrink: 0,
            flexWrap: 'wrap',
          }}>
            {allViolations.length === 0 ? (
              <span style={{ fontSize: 13, color: '#2e7d32', fontWeight: 700 }}>
                ✓ 違反なし
              </span>
            ) : (
              <>
                {errors.length > 0 && (
                  <span style={{
                    fontSize: 11, fontWeight: 700,
                    color: '#c62828', background: '#fdecea',
                    padding: '2px 10px', borderRadius: 12,
                    border: '1px solid #ef9a9a',
                  }}>
                    ✕ エラー {errors.length}件
                  </span>
                )}
                {warnings.length > 0 && (
                  <span style={{
                    fontSize: 11, fontWeight: 700,
                    color: '#f57f17', background: '#fff8e1',
                    padding: '2px 10px', borderRadius: 12,
                    border: '1px solid #ffe082',
                  }}>
                    ⚠ 警告 {warnings.length}件
                  </span>
                )}
              </>
            )}
            {constraintCheckedAt && (
              <span style={{ fontSize: 10, color: '#b0bec5', marginLeft: 'auto' }}>
                {new Date(constraintCheckedAt).toLocaleTimeString('ja-JP')}
              </span>
            )}
          </div>

          {/* Violation list */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {allViolations.length === 0 ? (
              <div style={{
                padding: 40, textAlign: 'center', color: '#90a4ae', fontSize: 13,
              }}>
                すべての制約チェックをクリアしました
              </div>
            ) : (
              <>
                {errors.length > 0 && (
                  <>
                    <div style={{
                      padding: '5px 12px', fontSize: 10, fontWeight: 700,
                      color: '#c62828', background: '#fdecea', letterSpacing: 1,
                    }}>
                      エラー
                    </div>
                    {errors.map((v, i) => (
                      <ViolationRow
                        key={i}
                        v={v}
                        isSelected={v.assignmentIndices[0] === state.selectedAssignmentIndex}
                        onClick={() => handleViolationClick(v)}
                      />
                    ))}
                  </>
                )}
                {warnings.length > 0 && (
                  <>
                    <div style={{
                      padding: '5px 12px', fontSize: 10, fontWeight: 700,
                      color: '#f57f17', background: '#fff8e1', letterSpacing: 1,
                    }}>
                      警告
                    </div>
                    {warnings.map((v, i) => (
                      <ViolationRow
                        key={i}
                        v={v}
                        isSelected={v.assignmentIndices[0] === state.selectedAssignmentIndex}
                        onClick={() => handleViolationClick(v)}
                      />
                    ))}
                  </>
                )}
              </>
            )}
          </div>

          {/* Footer */}
          <div style={{
            padding: '8px 14px',
            borderTop: '1px solid rgba(0,0,0,0.06)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            flexShrink: 0,
          }}>
            {allViolations.length > 0 && (
              <span style={{ fontSize: 10, color: '#b0bec5', flex: 1 }}>
                違反をクリックすると割付が選択されます
              </span>
            )}
            <button
              onClick={runCheck}
              disabled={isConstraintChecking}
              style={{
                marginLeft: 'auto',
                padding: '4px 14px',
                fontSize: 12,
                fontFamily: 'MS Gothic, monospace',
                background: isConstraintChecking ? '#bdbdbd' : '#1565c0',
                color: '#fff',
                border: 'none',
                borderRadius: 4,
                cursor: isConstraintChecking ? 'default' : 'pointer',
                flexShrink: 0,
              }}
            >
              {isConstraintChecking ? '⏳ チェック中...' : '☑ 再チェック'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
