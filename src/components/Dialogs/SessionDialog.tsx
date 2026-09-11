import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import {
  listSessions, getServerUrl, setServerUrl, fetchLanHosts, LanHost,
} from '../../services/collabService';
import { loadDisplayName, saveDisplayName } from '../../lib/collabPrefs';
import { SessionRole, SessionStatus, SessionSummary } from '../../types/appState';
import { UI } from '../../config/uiText';

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
};
const boxStyle: React.CSSProperties = {
  backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 560, width: '92%',
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
const titleStyle: React.CSSProperties = { fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 12 };

const PAGE_SIZE = 8;
// How long to let LAN discovery run before admitting "nothing found" — avoids
// flashing that message for the first ~1-2s while the beacon is still warming up.
const SEARCH_SETTLE_MS = 2500;

const STATUS_LABEL: Record<SessionStatus, string> = {
  open: UI.sessionStatusOpen, lock: UI.sessionStatusLock, close: UI.sessionStatusClose,
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

function formatJoinTime(ts: number | null): string {
  if (!ts) return UI.sessionListNeverJoined;
  return new Date(ts).toLocaleString('ja-JP', {
    month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

// ---- 参加 (session list + join) -------------------------------------------
// No server-address concept anywhere in this dialog. Finding the host is
// fully automatic (LAN discovery); if it can't find anything, it says so in
// plain language — there is no field for the user to fill in as a fallback.

function SessionJoinDialog({ onClose }: { onClose: () => void }) {
  const { joinCollabSession } = useAppContext();
  const [displayName, setDisplayName] = useState(() => loadDisplayName());
  const [role, setRole] = useState<SessionRole>('edit');
  const [target, setTarget] = useState(() => getServerUrl()); // '' = this PC
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [hosts, setHosts] = useState<LanHost[]>([]);
  const [searchSettled, setSearchSettled] = useState(false);
  const autoSwitchedRef = useRef(false);

  const refresh = useCallback(() => {
    listSessions()
      .then((rows) => { setSessions(rows); setError(null); })
      .catch(() => {
        // A remembered host (from a previous visit) that's no longer
        // reachable self-heals silently — fall back to this PC and let
        // discovery find something else, rather than dead-ending on an
        // error the user can't do anything about. Only a genuine failure to
        // reach this PC's own server is worth telling them about.
        setTarget((current) => {
          if (current === '') {
            setError(UI.sessionListUnreachableMessage);
            return current;
          }
          setServerUrl('');
          setSessions(null);
          setSelectedId(null);
          setPage(0);
          autoSwitchedRef.current = false;
          return '';
        });
      });
  }, []);

  // Point at a different server (auto-discovered or picked from the list —
  // there is no way for the user to type one) and reload from it.
  const switchTo = useCallback((url: string) => {
    setServerUrl(url);
    setTarget(url);
    setSessions(null);
    setSelectedId(null);
    setPage(0);
    setError(null);
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh, target]);

  useEffect(() => {
    let cancelled = false;
    const poll = () => void fetchLanHosts().then((found) => { if (!cancelled) setHosts(found); });
    poll();
    const interval = setInterval(poll, 4000);
    const settleTimer = setTimeout(() => { if (!cancelled) setSearchSettled(true); }, SEARCH_SETTLE_MS);
    return () => { cancelled = true; clearInterval(interval); clearTimeout(settleTimer); };
  }, []);

  // Silent hand-off: this PC has nothing to show and exactly one other app is
  // on the network — go straight there instead of telling the user "nothing
  // found" when there plainly is something. Zero clicks, zero typing.
  useEffect(() => {
    if (autoSwitchedRef.current || target !== '') return;
    if (sessions === null || sessions.length > 0) return;
    if (hosts.length === 1) {
      autoSwitchedRef.current = true;
      switchTo(hosts[0].url);
    }
  }, [target, sessions, hosts, switchTo]);

  // Most recently joined first; never-joined fall back to createdAt.
  const sorted = useMemo(
    () => [...(sessions ?? [])].sort((a, b) => (b.lastJoinAt ?? b.createdAt) - (a.lastJoinAt ?? a.createdAt)),
    [sessions],
  );
  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const pageRows = sorted.slice(clampedPage * PAGE_SIZE, clampedPage * PAGE_SIZE + PAGE_SIZE);
  const selectionOnPage = pageRows.some((r) => r.id === selectedId);
  // More than one other app found and there's nothing local to show — ask
  // which one, in plain names, instead of silently guessing or showing "empty".
  const showHostPicker = target === '' && sessions != null && sessions.length === 0 && hosts.length > 1;

  const handleJoin = async () => {
    if (!displayName.trim()) { setError(UI.sessionJoinNeedName); return; }
    if (!selectedId) { setError(UI.sessionJoinNeedSelection); return; }
    setBusy(true);
    setError(null);
    try {
      saveDisplayName(displayName);
      await joinCollabSession(selectedId, displayName.trim(), role);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={titleStyle}>{UI.sessionJoinDialogTitle}</div>

      <input
        placeholder={UI.sessionNamePlaceholder}
        value={displayName}
        onChange={(e) => setDisplayName(e.target.value)}
        style={{ ...inputStyle, marginBottom: 10 }}
      />
      <div style={{ display: 'flex', gap: 16, marginBottom: 10, fontSize: 12 }}>
        <label><input type="radio" checked={role === 'edit'} onChange={() => setRole('edit')} /> {UI.sessionJoinRoleEdit}</label>
        <label><input type="radio" checked={role === 'view'} onChange={() => setRole('view')} /> {UI.sessionJoinRoleView}</label>
      </div>

      {showHostPicker ? (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{UI.sessionPickHostLabel}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {hosts.map((h) => (
              <button
                key={h.url}
                onClick={() => switchTo(h.url)}
                style={{
                  fontSize: 12, padding: '4px 12px', borderRadius: 10, border: '1px solid #90caf9',
                  backgroundColor: '#e3f2fd', color: '#1565c0', cursor: 'pointer',
                }}
              >
                {h.name}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          {/* header */}
          <div style={{ display: 'flex', fontSize: 10, color: '#888', padding: '0 10px 2px' }}>
            <span style={{ flex: 1 }} />
            <span style={{ width: 64, textAlign: 'center' }} />
            <span style={{ width: 40, textAlign: 'right' }} />
            <span style={{ width: 96, textAlign: 'right' }}>{UI.sessionListLastJoinCol}</span>
          </div>
          <div style={{ border: '1px solid #e0e0e0', borderRadius: 2, height: PAGE_SIZE * 28, overflowY: 'auto' }}>
            {sessions == null && <div style={{ padding: 10, fontSize: 12, color: '#999' }}>...</div>}
            {sessions != null && sorted.length === 0 && (
              <div style={{ padding: 10, fontSize: 12, color: '#999' }}>
                {searchSettled ? UI.sessionNoneFoundMessage : UI.sessionSearchingMessage}
              </div>
            )}
            {pageRows.map((s) => (
              <div
                key={s.id}
                onClick={() => setSelectedId(s.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '5px 10px', fontSize: 12,
                  borderBottom: '1px solid #f0f0f0', cursor: 'pointer',
                  backgroundColor: s.id === selectedId ? '#e3f2fd' : undefined,
                }}
              >
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.name}</span>
                <span style={{ width: 64, textAlign: 'center' }}><StatusChip status={s.status} /></span>
                <span style={{ width: 40, textAlign: 'right', color: '#666' }}>{UI.sessionParticipantCount(s.participantCount)}</span>
                <span style={{ width: 96, textAlign: 'right', color: '#666', fontSize: 11 }}>{formatJoinTime(s.lastJoinAt)}</span>
              </div>
            ))}
          </div>

          {/* pager */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 6, fontSize: 12, color: '#666' }}>
            <button onClick={refresh} style={{ ...neutralBtnStyle, padding: '2px 10px', fontSize: 12 }}>{UI.sessionListRefreshBtn}</button>
            {sorted.length > PAGE_SIZE && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={clampedPage === 0}
                  style={{ ...neutralBtnStyle, padding: '2px 8px', fontSize: 12 }}
                >{UI.sessionListPrevBtn}</button>
                <span>{UI.sessionListPageLabel(clampedPage * PAGE_SIZE + 1, Math.min(sorted.length, (clampedPage + 1) * PAGE_SIZE), sorted.length)}</span>
                <button
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  disabled={clampedPage >= pageCount - 1}
                  style={{ ...neutralBtnStyle, padding: '2px 8px', fontSize: 12 }}
                >{UI.sessionListNextBtn}</button>
              </div>
            )}
          </div>
        </>
      )}

      {error && <div style={{ color: '#c62828', fontSize: 12, marginTop: 8 }}>{error}</div>}

      {/* select-then-join: the action button is at the bottom-right */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 14 }}>
        <button
          onClick={() => void handleJoin()}
          disabled={busy || !displayName.trim() || !selectionOnPage}
          style={primaryBtnStyle}
        >
          {UI.sessionJoinConfirmBtn}
        </button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}

// ---- 作成 (create) -------------------------------------------------------
// No server/network field here at all — creating always happens on this PC.

function SessionCreateDialog({ onClose }: { onClose: () => void }) {
  const { state, startCollabSession, createUploadSession } = useAppContext();
  const [displayName, setDisplayName] = useState(() => loadDisplayName());
  const [sessionName, setSessionName] = useState('');
  const [envFile, setEnvFile] = useState<File | null>(null);
  const [scheduleFile, setScheduleFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canUseCurrent = !!state.schedule && !!state.envConfig;

  const run = async (fn: () => Promise<unknown>) => {
    if (!displayName.trim()) { setError(UI.sessionJoinNeedName); return; }
    if (!sessionName.trim()) { setError(UI.sessionNameFieldPlaceholder); return; }
    setBusy(true);
    setError(null);
    try {
      saveDisplayName(displayName);
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
      <div style={titleStyle}>{UI.sessionCreateDialogTitle}</div>

      <input placeholder={UI.sessionNamePlaceholder} value={displayName} onChange={(e) => setDisplayName(e.target.value)} style={{ ...inputStyle, marginBottom: 10 }} />
      <input placeholder={UI.sessionNameFieldPlaceholder} value={sessionName} onChange={(e) => setSessionName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />

      <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>{UI.sessionCreateEnvFileLabel}</div>
      <input type="file" accept=".yaml,.yml" onChange={(e) => setEnvFile(e.target.files?.[0] ?? null)} style={{ marginBottom: 8, fontSize: 12 }} />
      <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>{UI.sessionCreateScheduleFileLabel}</div>
      <input type="file" accept=".yaml,.yml" onChange={(e) => setScheduleFile(e.target.files?.[0] ?? null)} style={{ marginBottom: 12, fontSize: 12 }} />

      {error && <div style={{ color: '#c62828', fontSize: 12, marginBottom: 8 }}>{error}</div>}

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        {canUseCurrent
          ? <button disabled={busy} onClick={() => void run(() => startCollabSession(displayName.trim(), sessionName.trim()))} style={neutralBtnStyle}>{UI.sessionCreateFromCurrentBtn}</button>
          : <span />}
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            disabled={busy || !envFile || !scheduleFile}
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

// ---- セッション情報 (info + shared lock) --------------------------------

function SessionInfoDialog({ onClose }: { onClose: () => void }) {
  const { state, lockSession, unlockSession, leaveCollabSession } = useAppContext();
  const session = state.session;

  if (!session) return null;

  return (
    <div>
      <div style={{ ...titleStyle, marginBottom: 4 }}>{UI.sessionActiveTitle}</div>
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
        {session.participants.map((p) => (
          <div key={p.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 10px', fontSize: 12, color: '#222' }}>
            <span>{p.name}</span>
            <span style={{ color: '#666', marginLeft: 12 }}>{p.role === 'edit' ? UI.sessionParticipantRoleEdit : UI.sessionParticipantRoleView}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        {/* Lock/unlock is a shared toggle — any participant may use it. */}
        {session.status === 'lock'
          ? <button onClick={() => unlockSession()} style={primaryBtnStyle}>{UI.sessionUnlockBtn}</button>
          : <button onClick={() => lockSession()} style={{ ...neutralBtnStyle, backgroundColor: '#e65100' }}>{UI.sessionLockBtn}</button>}
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => { leaveCollabSession(); onClose(); }} style={dangerBtnStyle}>{UI.sessionLeaveBtn}</button>
          <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
        </div>
      </div>
    </div>
  );
}

export function SessionDialog() {
  const { state, dispatch } = useAppContext();
  const kind = state.sessionDialog;
  if (!kind) return null;
  const handleClose = () => dispatch({ type: 'CLOSE_SESSION_DIALOG' });

  return (
    <div style={overlayStyle}>
      <div style={boxStyle}>
        {kind === 'join' && <SessionJoinDialog onClose={handleClose} />}
        {kind === 'create' && <SessionCreateDialog onClose={handleClose} />}
        {kind === 'info' && <SessionInfoDialog onClose={handleClose} />}
      </div>
    </div>
  );
}
