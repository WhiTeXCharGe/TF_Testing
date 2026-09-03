import { useState, useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import { fetchCollabLink, fetchSessionName, parseSessionId } from '../../services/collabService';
import { SessionRole } from '../../types/appState';
import { UI } from '../../config/uiText';

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
};
const boxStyle: React.CSSProperties = {
  backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 480, width: '90%',
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

function LinkRow({ label, link }: { label: string; link: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>{label}</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input readOnly value={link} onFocus={e => e.currentTarget.select()} style={{ ...inputStyle, flex: 1, backgroundColor: '#f8f9fa' }} />
        <button onClick={() => void handleCopy()} style={{ ...primaryBtnStyle, backgroundColor: copied ? '#388e3c' : '#1976d2', whiteSpace: 'nowrap' }}>
          {copied ? UI.copyLinkCopied : UI.copyLinkBtn}
        </button>
      </div>
    </div>
  );
}

function ActiveSessionPanel({ onClose }: { onClose: () => void }) {
  const { state, leaveCollabSession } = useAppContext();
  const [editLink, setEditLink] = useState<string | null>(null);
  const [viewLink, setViewLink] = useState<string | null>(null);
  const session = state.session;

  // Declared BEFORE the `if (!session)` guard below: hooks must run
  // unconditionally and in the same order on every render, so the early
  // return can never sit above a hook. That's why the deps and the body use
  // optional chaining — `session` may legitimately be null here.
  useEffect(() => {
    const sessionId = session?.id;
    if (!sessionId) return;
    let cancelled = false;
    if (session?.role === 'edit') {
      void fetchCollabLink(sessionId, 'edit').then(link => { if (!cancelled) setEditLink(link); }).catch(() => {});
    }
    void fetchCollabLink(sessionId, 'view').then(link => { if (!cancelled) setViewLink(link); }).catch(() => {});
    return () => { cancelled = true; };
  }, [session?.id, session?.role]);

  if (!session) return null;

  return (
    <div>
      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 4 }}>{UI.sessionActiveTitle}</div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 12 }}>{UI.sessionNameLabel}: {session.name}</div>
      {session.role === 'edit' && editLink && <LinkRow label={UI.sessionEditLinkLabel} link={editLink} />}
      {viewLink && <LinkRow label={UI.sessionViewLinkLabel} link={viewLink} />}
      <div style={{ fontSize: 11, color: '#666', marginBottom: 16 }}>
        {UI.sessionParticipantsLabel(session.participants.length)}
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <button onClick={() => { leaveCollabSession(); onClose(); }} style={dangerBtnStyle}>{UI.sessionLeaveBtn}</button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}

function StartOrJoinPanel({ onClose }: { onClose: () => void }) {
  const { state, startCollabSession, joinCollabSession } = useAppContext();
  const [tab, setTab] = useState<'start' | 'join'>(state.sessionDialogTab);
  const [displayName, setDisplayName] = useState('');
  const [sessionName, setSessionName] = useState('');
  const [joinInput, setJoinInput] = useState('');
  const [joinRole, setJoinRole] = useState<SessionRole>('edit');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resolvedSessionName, setResolvedSessionName] = useState<string | null>(null);
  const [nameLookupFailed, setNameLookupFailed] = useState(false);

  useEffect(() => {
    if (tab !== 'join' || !joinInput.trim()) {
      setResolvedSessionName(null);
      setNameLookupFailed(false);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      const id = parseSessionId(joinInput.trim());
      void fetchSessionName(id).then(name => {
        if (cancelled) return;
        setResolvedSessionName(name);
        setNameLookupFailed(!name);
      });
    }, 400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [tab, joinInput]);

  const handleStart = async () => {
    if (!displayName.trim() || !sessionName.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await startCollabSession(displayName.trim(), sessionName.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleJoin = async () => {
    if (!displayName.trim() || !joinInput.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await joinCollabSession(joinInput.trim(), displayName.trim(), joinRole);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button onClick={() => setTab('start')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'start' ? '#1976d2' : '#b0bec5' }}>{UI.startSessionItem}</button>
        <button onClick={() => setTab('join')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'join' ? '#1976d2' : '#b0bec5' }}>{UI.joinSessionItem}</button>
      </div>

      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 8 }}>
        {tab === 'start' ? UI.sessionDialogStartTitle : UI.sessionDialogJoinTitle}
      </div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
        {tab === 'start' ? UI.sessionDialogStartDesc : UI.sessionDialogJoinDesc}
      </div>

      {tab === 'start' && (
        <input placeholder={UI.sessionNameFieldPlaceholder} value={sessionName} onChange={e => setSessionName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />
      )}

      <input placeholder={UI.sessionNamePlaceholder} value={displayName} onChange={e => setDisplayName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />

      {tab === 'join' && (
        <>
          <input placeholder={UI.sessionJoinLinkPlaceholder} value={joinInput} onChange={e => setJoinInput(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />
          {resolvedSessionName && (
            <div style={{ fontSize: 12, color: '#1976d2', marginBottom: 12 }}>{UI.sessionJoinResolvedName(resolvedSessionName)}</div>
          )}
          {nameLookupFailed && (
            <div style={{ fontSize: 12, color: '#c62828', marginBottom: 12 }}>{UI.sessionJoinUnresolvedName}</div>
          )}
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 12 }}>
            <label><input type="radio" checked={joinRole === 'edit'} onChange={() => setJoinRole('edit')} /> {UI.sessionJoinRoleEdit}</label>
            <label><input type="radio" checked={joinRole === 'view'} onChange={() => setJoinRole('view')} /> {UI.sessionJoinRoleView}</label>
          </div>
        </>
      )}

      {error && <div style={{ color: '#c62828', fontSize: 12, marginBottom: 12 }}>{error}</div>}

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <button
          onClick={() => void (tab === 'start' ? handleStart() : handleJoin())}
          disabled={busy || !displayName.trim() || (tab === 'start' && !sessionName.trim()) || (tab === 'join' && !joinInput.trim())}
          style={primaryBtnStyle}
        >
          {tab === 'start' ? UI.sessionStartBtn : UI.sessionJoinBtn}
        </button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
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
