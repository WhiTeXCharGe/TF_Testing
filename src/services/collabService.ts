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
let socketOrigin: string | null = null;

// origin: explicit host to connect to (a joiner's parsed link) — falls back
// to this page's own host, which is only correct when the page itself was
// navigated to the host's URL (a plain browser join). The desktop app's
// window never navigates like that (it always loads its own embedded
// server), so joining a link pasted into its "Join Session" dialog MUST pass
// the link's host explicitly or it silently connects to the wrong machine.
// The port is always forced to 3010 (the API/socket server) regardless of
// the link's own port, which is the FRONTEND port — 5173 in dev, where no
// socket server listens; see socketOriginFromLink below.
function ensureSocket(origin?: string): Socket {
  const targetOrigin = origin ?? getSocketOrigin();
  if (socket && socketOrigin !== targetOrigin) {
    socket.disconnect();
    socket = null;
  }
  if (!socket) {
    socketOrigin = targetOrigin;
    socket = io(targetOrigin, {
      path: '/collab/socket.io',
      transports: ['websocket', 'polling'],
    });
  }
  return socket;
}

export async function createCollabSession(name: string, baseline: SessionBaseline): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/collab/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, ...baseline }),
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
  onSyncInit: (sessionName: string, baseline: SessionBaseline, actions: LoggedAction[]) => void,
  onAction: (action: { type: string; payload: unknown }) => void,
  onPresence: (participants: SessionParticipant[]) => void,
  onStatusChange: (status: SessionConnectionStatus) => void,
  origin?: string,
): () => void {
  const s = ensureSocket(origin);
  onStatusChange('connecting');

  // Consumed by the FIRST sync-init only. socket.io-client reconnects on its
  // own, and every reconnect re-emits 'join' and so gets another sync-init —
  // if the creator kept skipping those, they'd silently miss every action
  // that landed while they were disconnected while still reporting
  // 'connected'. From their second sync-init onward the creator catches up
  // via baseline + log replay exactly like every other participant.
  let skipBaselineReplay = isCreator;

  const handleConnect = () => s.emit('join', { sessionId, name, role });
  const handleSyncInit = (payload: { ok: boolean; name?: string; baseline?: SessionBaseline; actions?: LoggedAction[]; participants?: SessionParticipant[] }) => {
    if (!payload.ok || !payload.baseline || !payload.name) {
      onStatusChange('disconnected');
      return;
    }
    // On the very first sync-init the creator's local state already IS the
    // baseline (it was just POSTed from there) — re-applying it would
    // needlessly wipe their own undo history. Everyone else (a real joiner)
    // replays baseline + log to catch up.
    if (skipBaselineReplay) {
      skipBaselineReplay = false;
    } else {
      onSyncInit(payload.name, payload.baseline, payload.actions ?? []);
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
    socketOrigin = null;
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

export async function fetchSessionName(sessionId: string): Promise<string | null> {
  try {
    const res = await fetch(`${getSocketOrigin()}/api/collab/sessions/${sessionId}/name`);
    const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; name?: string };
    if (!res.ok || !data.ok || !data.name) return null;
    return data.name;
  } catch {
    return null;
  }
}

// Derives the API/socket origin from a full join link. Keeps the link's host
// but ALWAYS forces port 3010 — the link's own port is the frontend port
// (5173 in dev, 3010 in prod) and the socket/API server only ever listens on
// 3010. Returns null for a bare session id (no host to extract).
export function parseSessionOrigin(input: string): string | null {
  const trimmed = input.trim();
  try {
    const url = new URL(trimmed);
    return `${url.protocol}//${url.hostname}:3010`;
  } catch {
    return null;
  }
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