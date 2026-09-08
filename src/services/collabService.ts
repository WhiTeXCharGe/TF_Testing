import { io, Socket } from 'socket.io-client';
import type {
  SessionBaseline, SessionParticipant, SessionRole, SessionConnectionStatus,
  SessionStatus, SessionSummary,
} from '../types/appState';

export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

export interface CreateResult {
  sessionId: string;
  ownerToken: string;
}

export interface JoinCallbacks {
  onSyncInit: (sessionName: string, baseline: SessionBaseline, actions: LoggedAction[]) => void;
  onAction: (action: { type: string; payload: unknown }) => void;
  onPresence: (participants: SessionParticipant[]) => void;
  onStatusChange: (status: SessionConnectionStatus) => void;
  onSessionStatus: (status: SessionStatus) => void;
}

// Injected by Vite's `define` (see vite.config.ts) — a string literal at build
// time, absent under jest (guarded by `typeof`). Empty string in dev/LAN.
declare const __ACA1_URL__: string | undefined;

// ACA1 (session API) base URL. In dev/LAN it defaults to the current host on
// the mock's ACA1 port; a build sets VITE_ACA1_URL to the deployed ACA1.
function aca1Base(): string {
  const configured = typeof __ACA1_URL__ === 'string' ? __ACA1_URL__ : '';
  if (configured) return configured.replace(/\/+$/, '');
  return `${window.location.protocol}//${window.location.hostname}:4000`;
}

// ACA1 records its own reachable URL as PUBLIC_RELAY_URL. Locally that is
// http://localhost:4010, which is wrong for a LAN participant — rewrite the
// loopback host to whatever host actually loaded this page. A real Azure FQDN
// has no loopback host and is left untouched.
function rewriteLoopback(url: string): string {
  try {
    const u = new URL(url);
    if (u.hostname === 'localhost' || u.hostname === '127.0.0.1') {
      u.hostname = window.location.hostname;
    }
    return u.toString().replace(/\/+$/, '');
  } catch {
    return url;
  }
}

async function readJson(res: Response): Promise<Record<string, unknown>> {
  return (await res.json().catch(() => ({ ok: false }))) as Record<string, unknown>;
}

// ---- session lifecycle (HTTP to ACA1) -------------------------------------

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${aca1Base()}/api/sessions`);
  const data = await readJson(res);
  if (!res.ok || !data.ok || !Array.isArray(data.sessions)) {
    throw new Error((data.error as string) ?? 'セッション一覧の取得に失敗しました');
  }
  return data.sessions as SessionSummary[];
}

export async function createSessionFromState(name: string, baseline: SessionBaseline): Promise<CreateResult> {
  const res = await fetch(`${aca1Base()}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, ...baseline }),
  });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.sessionId) throw new Error((data.error as string) ?? 'セッションの作成に失敗しました');
  return { sessionId: data.sessionId as string, ownerToken: data.ownerToken as string };
}

export async function createSessionFromYaml(
  name: string, scheduleFile: File, envConfigFile: File,
): Promise<CreateResult> {
  const form = new FormData();
  form.append('name', name);
  form.append('schedule', scheduleFile);
  form.append('envConfig', envConfigFile);
  const res = await fetch(`${aca1Base()}/api/sessions`, { method: 'POST', body: form });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.sessionId) throw new Error((data.error as string) ?? 'セッションの作成に失敗しました');
  return { sessionId: data.sessionId as string, ownerToken: data.ownerToken as string };
}

export async function openSession(sessionId: string): Promise<{ relayUrl: string; status: SessionStatus }> {
  const res = await fetch(`${aca1Base()}/api/sessions/${encodeURIComponent(sessionId)}/open`, { method: 'POST' });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.relayUrl) throw new Error((data.error as string) ?? 'セッションを開けませんでした');
  return { relayUrl: rewriteLoopback(data.relayUrl as string), status: data.status as SessionStatus };
}

export async function deleteSession(sessionId: string, ownerToken: string): Promise<void> {
  const res = await fetch(`${aca1Base()}/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: { 'x-owner-token': ownerToken },
  });
  if (!res.ok) {
    const data = await readJson(res);
    throw new Error((data.error as string) ?? 'セッションの削除に失敗しました');
  }
}

export async function fetchSessionName(sessionId: string): Promise<string | null> {
  try {
    const res = await fetch(`${aca1Base()}/api/sessions/${encodeURIComponent(sessionId)}`);
    const data = await readJson(res);
    const session = data.session as { name?: string } | undefined;
    if (!res.ok || !data.ok || !session?.name) return null;
    return session.name;
  } catch {
    return null;
  }
}

// ---- live relay (Socket.IO to ACA2) -------------------------------------

let socket: Socket | null = null;
let socketOrigin: string | null = null;

function ensureSocket(relayUrl: string): Socket {
  if (socket && socketOrigin !== relayUrl) {
    socket.disconnect();
    socket = null;
  }
  if (!socket) {
    socketOrigin = relayUrl;
    socket = io(relayUrl, { path: '/collab/socket.io', transports: ['websocket', 'polling'] });
  }
  return socket;
}

export function joinCollabRoom(
  sessionId: string,
  name: string,
  role: SessionRole,
  isCreator: boolean,
  relayUrl: string,
  ownerToken: string | undefined,
  cb: JoinCallbacks,
): () => void {
  const s = ensureSocket(relayUrl);
  cb.onStatusChange('connecting');

  // Consumed by the FIRST sync-init only. socket.io-client reconnects on its
  // own and re-emits 'join' on every reconnect; from the creator's second
  // sync-init onward they catch up via baseline + log replay like everyone
  // else (otherwise they would silently miss edits made while disconnected).
  let skipBaselineReplay = isCreator;

  const handleConnect = () => s.emit('join', { sessionId, name, role, ownerToken });
  const handleSyncInit = (payload: {
    ok: boolean; name?: string; baseline?: SessionBaseline; actions?: LoggedAction[];
    participants?: SessionParticipant[]; status?: SessionStatus;
  }) => {
    if (!payload.ok || !payload.baseline || !payload.name) {
      cb.onStatusChange('disconnected');
      return;
    }
    if (payload.status) cb.onSessionStatus(payload.status);
    if (skipBaselineReplay) {
      skipBaselineReplay = false;
    } else {
      cb.onSyncInit(payload.name, payload.baseline, payload.actions ?? []);
    }
    cb.onPresence(payload.participants ?? []);
    cb.onStatusChange('connected');
  };
  const handleAction = (payload: { type: string; payload: unknown }) => cb.onAction(payload);
  const handlePresence = (participants: SessionParticipant[]) => cb.onPresence(participants);
  const handleSessionStatus = (payload: { status: SessionStatus }) => cb.onSessionStatus(payload.status);
  const handleDisconnect = () => cb.onStatusChange('disconnected');

  s.on('connect', handleConnect);
  s.on('sync-init', handleSyncInit);
  s.on('action', handleAction);
  s.on('presence', handlePresence);
  s.on('session-status', handleSessionStatus);
  s.on('disconnect', handleDisconnect);

  if (s.connected) handleConnect();

  return () => {
    s.off('connect', handleConnect);
    s.off('sync-init', handleSyncInit);
    s.off('action', handleAction);
    s.off('presence', handlePresence);
    s.off('session-status', handleSessionStatus);
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

export function sendCollabLock(): void {
  socket?.emit('lock');
}

export function sendCollabUnlock(): void {
  socket?.emit('unlock');
}

// Accepts a bare session id or a full link (?session=<id>) pasted anywhere.
export function parseSessionId(input: string): string {
  const trimmed = input.trim();
  try {
    return new URL(trimmed).searchParams.get('session') ?? trimmed;
  } catch {
    return trimmed;
  }
}
