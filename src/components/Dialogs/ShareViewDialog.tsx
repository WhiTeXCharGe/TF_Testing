import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { stopHostBroadcast } from '../../services/viewBroadcastService';
import { UI } from '../../config/uiText';

export function ShareViewDialog() {
  const { state, dispatch } = useAppContext();
  const [copied, setCopied] = useState(false);

  if (!state.isShareViewDialogOpen) return null;

  const handleClose = () => dispatch({ type: 'CLOSE_SHARE_VIEW_DIALOG' });

  const handleStop = () => {
    stopHostBroadcast();
    dispatch({ type: 'SET_SHARING_LIVE_VIEW', payload: false });
    dispatch({ type: 'SET_LIVE_VIEW_SHARE_LINK', payload: null });
    handleClose();
  };

  const handleCopy = async () => {
    if (!state.liveViewShareLink) return;
    await navigator.clipboard.writeText(state.liveViewShareLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}
    >
      <div
        style={{
          backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 480, width: '90%',
          boxShadow: '0 4px 16px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 8 }}>
          {UI.shareViewDialogTitle}
        </div>
        <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
          {UI.shareViewDialogDesc}
        </div>

        {state.liveViewShareLink ? (
          <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
            <input
              readOnly
              value={state.liveViewShareLink}
              onFocus={(e) => e.currentTarget.select()}
              style={{
                flex: 1, padding: '6px 8px', fontSize: 12, fontFamily: 'monospace',
                border: '1px solid #ccc', borderRadius: 4, backgroundColor: '#f8f9fa',
              }}
            />
            <button
              onClick={handleCopy}
              style={{
                padding: '6px 14px', backgroundColor: copied ? '#388e3c' : '#1976d2', color: '#fff',
                border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12, whiteSpace: 'nowrap',
              }}
            >
              {copied ? UI.copyLinkCopied : UI.copyLinkBtn}
            </button>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#999', marginBottom: 20 }}>{UI.shareViewLinkLoading}</div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button
            onClick={handleStop}
            style={{
              padding: '6px 16px', backgroundColor: '#c62828', color: '#fff',
              border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
            }}
          >
            {UI.stopSharingBtn}
          </button>
          <button
            onClick={handleClose}
            style={{
              padding: '6px 16px', backgroundColor: '#78909c', color: '#fff',
              border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
            }}
          >
            {UI.shareViewClose}
          </button>
        </div>
      </div>
    </div>
  );
}
