import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

const btn: CSSProperties = {
  padding: '4px 10px',
  backgroundColor: '#fff',
  border: '1px solid #999',
  borderRadius: 3,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
};

export function UndoRedoButtons() {
  const { state, dispatch } = useAppContext();
  const isReadOnly = state.session?.role === 'view';
  const canUndo = state.undoStack.length > 0 && !isReadOnly;
  const canRedo = state.redoStack.length > 0 && !isReadOnly;

  return (
    <div style={{ display: 'flex', gap: 4 }}>
      <button
        onClick={() => dispatch({ type: 'UNDO' })}
        disabled={!canUndo}
        style={{ ...btn, opacity: canUndo ? 1 : 0.4 }}
      >
        {UI.undo}
      </button>
      <button
        onClick={() => dispatch({ type: 'REDO' })}
        disabled={!canRedo}
        style={{ ...btn, opacity: canRedo ? 1 : 0.4 }}
      >
        {UI.redo}
      </button>
    </div>
  );
}
