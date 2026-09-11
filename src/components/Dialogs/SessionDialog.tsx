import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import {
  listSessions, getServerUrl, setServerUrl, fetchLanAddresses, fetchLanHosts, LanHost,
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

// Which server (desktop-app host) to talk to. Blank = this app's own server.
// Discovered LAN hosts (see lan/discoveryBeacon.ts on the server) render as
// clickable chips so a joiner never has to type an address by hand; typing
// stays available as a manual fallback (different subnet, firewalled, ...).
function ServerUrlField({ onApply, autoSelectSingle = false }: { onApply?: () => void; autoSelectSingle?: boolean }) {
  const [value, setValue] = useState(() => getServerUrl());
  const [hosts, setHosts] = useState<LanHost[]>([]);
  const autoAppliedRef = useRef(false);

  const apply = useCallback((url: string) => {
    setValue(url);
    setServerUrl(url);
    onApply?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = () => void fetchLanHosts().then((found) => { if (!cancelled) setHosts(found); });
    poll();
    const timer = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  // Zero-click case: exactly one other app found on the LAN and the field is
  // still untouched (blank) — go straight there instead of making the user
  // click. Only fires once, and only where join semantics want it.
  useEffect(() => {
    if (!autoSelectSingle || autoAppliedRef.current) return;
    if (value === '' && hosts.length === 1) {
      autoAppliedRef.current = true;
      apply(hosts[0].url);
    }
  }, [autoSelectSingle, value, hosts, apply]);

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>{UI.sessionServerUrlLabel}</div>
      <input
        placeholder={UI.sessionServerUrlPlaceholder}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={() => apply(value)}
        onKeyDown={(e) => { if (e.key === 'Enter') apply(value); }}
        style={inputStyle}
      />
      {hosts.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, marginTop: 6 }}>
          <span style={{ fontSize: 11, color: '#666' }}>{UI.sessionDiscoveredLabel}</span>
          {hosts.map((h) => (
            <button
              key={h.url}
              onClick={() => apply(h.url)}
              style={{
                fontSize: 11, padding: '2px 10px', borderRadius: 10, border: '1px solid #90caf9', cursor: 'pointer',
                backgroundColor: h.url === value ? '#1976d2' : '#e3f2fd',
                color: h.url === value ? '#fff' : '#1565c0',
              }}
            >
              {h.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- 参加 (session list + join) -------------------------------------------

function SessionJoinDialog({ onClose }: { onClose: () => void }) {
  const { joinCollabSession } = useAppContext();
  const [displayName, setDisplayName] = useState(() => loadDisplayName());
  const [role, setRole] = useState<SessionRole>('edit');
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    listSessions()
      .then((rows) => {
        setSessions(rows);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  // Most recently joined first; never-joined fall back to createdAt.
  const sorted = useMemo(
    () => [...(sessions ?? [])].sort((a, b) => (b.lastJoinAt ?? b.createdAt) - (a.lastJoinAt ?? a.createdAt)),
    [sessions],
  );
  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const pageRows = sorted.slice(clampedPage * PAGE_SIZE, clampedPage * PAGE_SIZE + PAGE_SIZE);
  const selectionOnPage = pageRows.some((r) => r.id === selectedId);

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

      <ServerUrlField
        autoSelectSingle
        onApply={() => { setSessions(null); setSelectedId(null); setPage(0); refresh(); }}
      />

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
          <div style={{ padding: 10, fontSize: 12, color: '#999' }}>{UI.sessionListEmpty}</div>
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

      <ServerUrlField />
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
  const [lanAddrs, setLanAddrs] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    void fetchLanAddresses().then((a) => { if (!cancelled) setLanAddrs(a); });
    return () => { cancelled = true; };
  }, []);

  if (!session) return null;

  return (
    <div>
      <div style={{ ...titleStyle, marginBottom: 4 }}>{UI.sessionActiveTitle}</div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
        <span>{UI.sessionNameLabel}: {session.name}</span>
        <StatusChip status={session.status} />
      </div>

      {lanAddrs.length > 0 && (
        <div style={{ fontSize: 11, color: '#555', backgroundColor: '#f1f8e9', border: '1px solid #c5e1a5', borderRadius: 4, padding: '6px 10px', marginBottom: 12 }}>
          <div style={{ marginBottom: 2 }}>{UI.sessionJoinAddressHint}</div>
          {lanAddrs.map((ip) => (
            <div key={ip} style={{ fontFamily: 'monospace' }}>{`http://${ip}:${window.location.port || '3010'}`}</div>
          ))}
        </div>
      )}

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
