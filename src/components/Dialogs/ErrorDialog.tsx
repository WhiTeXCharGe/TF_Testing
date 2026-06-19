import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

export function ErrorDialog() {
  const { state, dispatch } = useAppContext();
  if (!state.errorMessage) return null;

  return (
    <div style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
    }}>
      <div style={{
        backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 420, width: '90%',
        boxShadow: '0 4px 16px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
      }}>
        <div style={{ fontSize: 15, fontWeight: 'bold', color: '#c62828', marginBottom: 12 }}>{UI.errorTitle}</div>
        <div style={{ fontSize: 13, color: '#333', marginBottom: 20, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {state.errorMessage}
        </div>
        <button
          onClick={() => dispatch({ type: 'SET_ERROR', payload: null })}
          style={{
            padding: '6px 20px', backgroundColor: '#1976d2', color: '#fff',
            border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
          }}
        >
          {UI.errorClose}
        </button>
      </div>
    </div>
  );
}
