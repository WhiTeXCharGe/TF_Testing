import { useState, useEffect, useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';
import { listSessions } from '../../services/collabService';
import { SessionRole, SessionStatus, SessionSummary } from '../../types/appState';
import { UI } from '../../config/uiText';

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
};
const boxStyle: React.CSSProperties = {
  backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 520, width: '90%',
  boxShadow: '0 4px 16px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
};
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '6px 8px', fontSize: 12, border: '1px solid #ccc', borderRadius: 4, boxSizing: 'border-box',
};
const primaryBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#1976d2', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};
const dangerBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#c62828', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};
const neutralBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#78909c', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};

const STATUS_LABEL: Record<SessionStatus, string> = {
  open: UI.sessionStatusOpen,
  lock: UI.sessionStatusLock,
  close: UI.sessionStatusClose,
};
const STATUS_COLOR: Record<SessionStatus, string> = { open: '#2e7d32', lock: '#e65100', close: '#757575' };

function StatusChip({ status }: { status: SessionStatus }) {
  return (
    <span style={{
      fontSize: 11, color: '#fff', backgroundColor: STATUS_COLOR[status],
      borderRadius: 10, padding: '1px 8px', whiteSpace: 'nowrap',
    }}>
      {STATUS_LABEL[status]}
    </span>
  );
}

function ActiveSessionPanel({ onClose }: { onClose: () => void }) {
  const { state, lockSession, unlockSession, leaveCollabSession } = useAppContext();
  const session = state.session;
  if (!session) return null;
  const isOwner = !!session.ownerToken;

  return (
    <div>
      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 4 }}>{UI.sessionActiveTitle}</div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
        <span>{UI.sessionNameLabel}: {session.name}</span>
        <StatusChip status={session.status} />
      </div>

      {session.status === 'lock' && (
        <div style={{ fontSize: 12, color: '#e65100', backgroundColor: '#fff3e0', border: '1px solid #ffcc80', borderRadius: 4, padding: '6px 10px', marginBottom: 12 }}>
          {UI.sessionLockedBannerText}
        </div>
      )}

      <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>
        {UI.sessionParticipantsLabel(session.participants.length)}
      </div>
      <div style={{ marginBottom: 16, border: '1px solid #e0e0e0', borderRadius: 2 }}>
        {session.participants.map(p => (
          <div key={p.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 10px', fontSize: 12, color: '#222' }}>
            <span>{p.name}</span>
            <span style={{ color: '#666', marginLeft: 12 }}>{p.role === 'edit' ? UI.sessionParticipantRoleEdit : UI.sessionParticipantRoleView}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <div>
          {isOwner && (session.status === 'lock'
            ? <button onClick={() => unlockSession()} style={primaryBtnStyle}>{UI.sessionUnlockBtn}</button>
            : <button onClick={() => lockSession()} style={{ ...neutralBtnStyle, backgroundColor: '#e65100' }}>{UI.sessionLockBtn}</button>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => { leaveCollabSession(); onClose(); }} style={dangerBtnStyle}>{UI.sessionLeaveBtn}</button>
          <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
        </div>
      </div>
    </div>
  );
}

function SessionListTab({ displayName, role, onClose }: { displayName: string; role: SessionRole; onClose: () => void }) {
  const { joinCollabSession } = useAppContext();
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openingId, setOpeningId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listSessions().then(setSessions).catch(err => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleOpen = async (id: string) => {
    if (!displayName.trim()) { setError(UI.joinSessionNamePlaceholder); return; }
    setOpeningId(id);
    setError(null);
    try {
      await joinCollabSession(id, displayName.trim(), role);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOpeningId(null);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 'bold', color: '#1a2e3f' }}>{UI.sessionListTab}</div>
        <button onClick={refresh} style={{ ...neutralBtnStyle, padding: '2px 10px', fontSize: 12 }}>{UI.sessionListRefreshBtn}</button>
      </div>

      <div style={{ border: '1px solid #e0e0e0', borderRadius: 2, marginBottom: 12, maxHeight: 220, overflowY: 'auto' }}>
        {sessions == null && <div style={{ padding: 10, fontSize: 12, color: '#999' }}>...</div>}
        {sessions != null && sessions.length === 0 && (
          <div style={{ padding: 10, fontSize: 12, color: '#999' }}>{UI.sessionListEmpty}</div>
        )}
        {sessions?.map(s => (
          <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderBottom: '1px solid #f0f0f0', fontSize: 12 }}>
            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.name}</span>
            <StatusChip status={s.status} />
            <span style={{ color: '#666', minWidth: 32, textAlign: 'right' }}>{UI.sessionParticipantCount(s.participantCount)}</span>
            <button
              onClick={() => void handleOpen(s.id)}
              disabled={openingId != null}
              style={{ ...primaryBtnStyle, padding: '3px 12px', fontSize: 12 }}
            >
              {UI.sessionListOpenBtn}
            </button>
          </div>
        ))}
      </div>

      {error && <div style={{ color: '#c62828', fontSize: 12, marginBottom: 8 }}>{error}</div>}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}

function SessionCreateTab({ displayName, onClose }: { displayName: string; onClose: () => void }) {
  const { state, startCollabSession, createUploadSession } = useAppContext();
  const [sessionName, setSessionName] = useState('');
  const [scheduleFile, setScheduleFile] = useState<File | null>(null);
  const [envFile, setEnvFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canUseCurrent = !!state.schedule && !!state.envConfig;

  const run = async (fn: () => Promise<unknown>) => {
    if (!displayName.trim() || !sessionName.trim()) { setError(UI.sessionNameFieldPlaceholder); return; }
    setBusy(true);
    setError(null);
    try {
      await fn();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 8 }}>{UI.sessionCreateTab}</div>

      <input placeholder={UI.sessionNameFieldPlaceholder} value={sessionName} onChange={e => setSessionName(e.target.value)} style={{ ...inputStyle, marginBottom: 10 }} />

      <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>{UI.sessionCreateScheduleFileLabel}</div>
      <input type="file" accept=".yaml,.yml" onChange={e => setScheduleFile(e.target.files?.[0] ?? null)} style={{ marginBottom: 8, fontSize: 12 }} />
      <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>{UI.sessionCreateEnvFileLabel}</div>
      <input type="file" accept=".yaml,.yml" onChange={e => setEnvFile(e.target.files?.[0] ?? null)} style={{ marginBottom: 12, fontSize: 12 }} />

      {error && <div style={{ color: '#c62828', fontSize: 12, marginBottom: 8 }}>{error}</div>}

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        {canUseCurrent
          ? <button disabled={busy} onClick={() => void run(() => startCollabSession(displayName.trim(), sessionName.trim()))} style={neutralBtnStyle}>{UI.sessionCreateFromCurrentBtn}</button>
          : <span />}
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            disabled={busy || !scheduleFile || !envFile}
            onClick={() => void run(() => createUploadSession(displayName.trim(), sessionName.trim(), scheduleFile!, envFile!))}
            style={primaryBtnStyle}
          >
            {UI.sessionCreateSubmitBtn}
          </button>
          <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
        </div>
      </div>
    </div>
  );
}

function StartOrJoinPanel({ onClose }: { onClose: () => void }) {
  const { state } = useAppContext();
  const [tab, setTab] = useState<'list' | 'create'>(state.sessionDialogTab);
  const [displayName, setDisplayName] = useState('');
  const [role, setRole] = useState<SessionRole>('edit');

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button onClick={() => setTab('list')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'list' ? '#1976d2' : '#b0bec5' }}>{UI.sessionListTab}</button>
        <button onClick={() => setTab('create')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'create' ? '#1976d2' : '#b0bec5' }}>{UI.sessionCreateTab}</button>
      </div>

      <input placeholder={UI.sessionNamePlaceholder} value={displayName} onChange={e => setDisplayName(e.target.value)} style={{ ...inputStyle, marginBottom: 10 }} />

      {tab === 'list' && (
        <div style={{ display: 'flex', gap: 16, marginBottom: 12, fontSize: 12 }}>
          <label><input type="radio" checked={role === 'edit'} onChange={() => setRole('edit')} /> {UI.sessionJoinRoleEdit}</label>
          <label><input type="radio" checked={role === 'view'} onChange={() => setRole('view')} /> {UI.sessionJoinRoleView}</label>
        </div>
      )}

      {tab === 'list'
        ? <SessionListTab displayName={displayName} role={role} onClose={onClose} />
        : <SessionCreateTab displayName={displayName} onClose={onClose} />}
    </div>
  );
}

export function SessionDialog() {
  const { state, dispatch } = useAppContext();
  if (!state.isSessionDialogOpen) return null;
  const handleClose = () => dispatch({ type: 'CLOSE_SESSION_DIALOG' });

  return (
    <div style={overlayStyle}>
      <div style={boxStyle}>
        {state.session ? <ActiveSessionPanel onClose={handleClose} /> : <StartOrJoinPanel onClose={handleClose} />}
      </div>
    </div>
  );
}
