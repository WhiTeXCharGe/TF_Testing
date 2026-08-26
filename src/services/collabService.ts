import { io, Socket } from 'socket.io-client';
import type { SessionBaseline, SessionParticipant, SessionRole, SessionConnectionStatus } from '../types/appState';

export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

// Same-machine relay: the API/socket server always runs on port 3010 on the
// same machine as the frontend, whether reached as localhost (host) or a LAN
// IP (a joiner who opened the shared link) — window.location.hostname already
// reflects whichever address was actually used to load the page.
function getSocketOrigin(): string {
  return `${window.location.protocol}//${window.location.hostname}:3010`;
}

let socket: Socket | null = null;

function ensureSocket(): Socket {
  if (!socket) {
    socket = io(getSocketOrigin(), {
      path: '/collab/socket.io',
      transports: ['websocket', 'polling'],
    });
  }
  return socket;
}

export async function createCollabSession(baseline: SessionBaseline): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/collab/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(baseline),
  });
  const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; sessionId?: string; error?: string };
  if (!res.ok || !data.ok || !data.sessionId) throw new Error(data.error ?? 'セッションの作成に失敗しました');
  return data.sessionId;
}

export function joinCollabRoom(
  sessionId: string,
  name: string,
  role: SessionRole,
  isCreator: boolean,
  onSyncInit: (baseline: SessionBaseline, actions: LoggedAction[]) => void,
  onAction: (action: { type: string; payload: unknown }) => void,
  onPresence: (participants: SessionParticipant[]) => void,
  onStatusChange: (status: SessionConnectionStatus) => void,
): () => void {
  const s = ensureSocket();
  onStatusChange('connecting');

  const handleConnect = () => s.emit('join', { sessionId, name, role });
  const handleSyncInit = (payload: { ok: boolean; baseline?: SessionBaseline; actions?: LoggedAction[]; participants?: SessionParticipant[] }) => {
    if (!payload.ok || !payload.baseline) {
      onStatusChange('disconnected');
      return;
    }
    // The creator's local state already IS the baseline (it was just POSTed
    // from there) — re-applying it would needlessly wipe their own undo
    // history. Everyone else (a real joiner) replays baseline + log to catch up.
    if (!isCreator) {
      onSyncInit(payload.baseline, payload.actions ?? []);
    }
    onPresence(payload.participants ?? []);
    onStatusChange('connected');
  };
  const handleAction = (payload: { type: string; payload: unknown }) => onAction(payload);
  const handlePresence = (participants: SessionParticipant[]) => onPresence(participants);
  const handleDisconnect = () => onStatusChange('disconnected');

  s.on('connect', handleConnect);
  s.on('sync-init', handleSyncInit);
  s.on('action', handleAction);
  s.on('presence', handlePresence);
  s.on('disconnect', handleDisconnect);

  if (s.connected) handleConnect();

  return () => {
    s.off('connect', handleConnect);
    s.off('sync-init', handleSyncInit);
    s.off('action', handleAction);
    s.off('presence', handlePresence);
    s.off('disconnect', handleDisconnect);
    s.emit('leave');
    s.disconnect();
    socket = null;
  };
}

export function sendCollabAction(type: string, payload: unknown): void {
  socket?.emit('action', { type, payload });
}

// Builds a link a participant on another PC (or the same PC, another tab) can
// open directly, e.g. http://192.168.1.23:5173/?session=<id>&role=edit —
// reuses whatever port/path this client is currently on.
export async function fetchCollabLink(sessionId: string, role: SessionRole): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/network-info`);
  const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; addresses?: string[]; error?: string };
  if (!res.ok || !data.ok || !data.addresses) throw new Error(data.error ?? 'ネットワーク情報の取得に失敗しました');
  const lanIp = data.addresses[0] ?? window.location.hostname;
  return `${window.location.protocol}//${lanIp}:${window.location.port}${window.location.pathname}?session=${sessionId}&role=${role}`;
}

// Accepts either a bare session id or a full link (as produced by
// fetchCollabLink above) pasted into the "Join Session" dialog.
export function parseSessionId(input: string): string {
  const trimmed = input.trim();
  try {
    const url = new URL(trimmed);
    return url.searchParams.get('session') ?? trimmed;
  } catch {
    return trimmed;
  }
}
