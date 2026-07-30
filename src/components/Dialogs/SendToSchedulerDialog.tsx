import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { downloadBothYamlFiles } from '../../services/fileService';
import { sendToScheduler } from '../../services/handoffService';
import { UI } from '../../config/uiText';

type Stage = 'confirm' | 'sending' | 'done' | 'error';

export function SendToSchedulerDialog() {
  const { state, dispatch } = useAppContext();
  const [stage, setStage] = useState<Stage>('confirm');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  if (!state.isSendToSchedulerDialogOpen) return null;
  if (!state.envConfig || !state.schedule) return null;

  const close = () => {
    dispatch({ type: 'CLOSE_SEND_TO_SCHEDULER_DIALOG' });
    // Reset for next time the dialog is opened.
    setStage('confirm');
    setErrorMsg(null);
    setResultUrl(null);
  };

  const handleConfirm = async () => {
    if (!state.envConfig || !state.schedule) return;
    setStage('sending');
    try {
      downloadBothYamlFiles(
        state.envConfig,
        state.schedule,
        state.currentEnvPath ?? 'EnvConfig.yaml',
        state.currentSchedulePath ?? 'Schedule.yaml',
      );
      const { url } = await sendToScheduler(state.envConfig, state.schedule);
      // url is null in desktop mode — the handoff was already delivered
      // straight to SchedulerWeb's window, nothing left to open here.
      if (url) {
        const opened = window.open(url, '_blank', 'noopener');
        setResultUrl(opened ? null : url);
      } else {
        setResultUrl(null);
      }
      setStage('done');
    } catch (e) {
      setErrorMsg(String((e as Error).message ?? e));
      setStage('error');
    }
  };

  const canClose = stage === 'confirm' || stage === 'done' || stage === 'error';

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 1200,
        background: 'rgba(0,0,0,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={e => { if (e.target === e.currentTarget && canClose) close(); }}
    >
      <div style={{
        background: '#fff', borderRadius: 6, padding: 24, width: 440,
        boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
        fontFamily: 'Meiryo, sans-serif',
      }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, color: '#1c2b3a' }}>
          {UI.sendToSchedulerDialogTitle}
        </div>

        {stage === 'confirm' && (
          <>
            <div style={{ fontSize: 13, color: '#333', marginBottom: 16, lineHeight: 1.6 }}>
              {UI.sendToSchedulerConfirmBody}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={close}
                style={{
                  padding: '6px 18px', border: '1px solid #c0d0e0', borderRadius: 4,
                  background: '#f5f8fc', cursor: 'pointer', fontSize: 13,
                  fontFamily: 'Meiryo, sans-serif', color: '#333',
                }}
              >
                {UI.dialogCancel}
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                style={{
                  padding: '6px 18px', border: 'none', borderRadius: 4,
                  background: '#1c4f8a', color: '#fff', cursor: 'pointer', fontSize: 13,
                  fontFamily: 'Meiryo, sans-serif',
                }}
              >
                {UI.sendBtn}
              </button>
            </div>
          </>
        )}

        {stage === 'sending' && (
          <div style={{ fontSize: 13, color: '#333', padding: '8px 0' }}>
            {UI.sendingStatus}
          </div>
        )}

        {stage === 'done' && (
          <>
            <div style={{ fontSize: 13, color: '#2e7d32', marginBottom: 12 }}>
              {UI.sendDoneMessage}
            </div>
            {resultUrl && (
              <div style={{ fontSize: 12, color: '#333', marginBottom: 16, wordBreak: 'break-all' }}>
                {UI.sendDoneManualLinkHint}{' '}
                <a href={resultUrl} target="_blank" rel="noopener noreferrer">{resultUrl}</a>
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={close}
                style={{
                  padding: '6px 18px', border: 'none', borderRadius: 4,
                  background: '#1c4f8a', color: '#fff', cursor: 'pointer', fontSize: 13,
                  fontFamily: 'Meiryo, sans-serif',
                }}
              >
                {UI.closeBtn}
              </button>
            </div>
          </>
        )}

        {stage === 'error' && (
          <>
            <div style={{
              background: '#fdecea', color: '#c62828', border: '1px solid #f5c6cb',
              borderRadius: 4, padding: '8px 12px', fontSize: 12, marginBottom: 16,
              whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            }}>
              {errorMsg}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={close}
                style={{
                  padding: '6px 18px', border: '1px solid #c0d0e0', borderRadius: 4,
                  background: '#f5f8fc', cursor: 'pointer', fontSize: 13,
                  fontFamily: 'Meiryo, sans-serif', color: '#333',
                }}
              >
                {UI.closeBtn}
              </button>
              <button
                type="button"
                onClick={() => setStage('confirm')}
                style={{
                  padding: '6px 18px', border: 'none', borderRadius: 4,
                  background: '#1c4f8a', color: '#fff', cursor: 'pointer', fontSize: 13,
                  fontFamily: 'Meiryo, sans-serif',
                }}
              >
                {UI.retryBtn}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}