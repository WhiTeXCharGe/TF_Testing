import { io, Socket } from 'socket.io-client';
import type {
  SessionBaseline, SessionParticipant, SessionRole, SessionConnectionStatus,
  SessionStatus, SessionSummary,
} from '../types/appState';
import { parseScheduleYaml, parseEnvConfigYaml } from './yamlService';

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

// Runtime override: the desktop app / LAN host address a participant types in
// the join dialog. Blank = talk to this app's own bundled server (the
// packaged Electron window loads it from the same origin).
const SERVER_URL_KEY = 'gantt.collab.serverUrl';

export function getServerUrl(): string {
  try {
    return localStorage.getItem(SERVER_URL_KEY) ?? '';
  } catch {
    return '';
  }
}

export function setServerUrl(url: string): void {
  try {
    const trimmed = url.trim().replace(/\/+$/, '');
    if (trimmed) localStorage.setItem(SERVER_URL_KEY, trimmed);
    else localStorage.removeItem(SERVER_URL_KEY);
  } catch {
    /* storage disabled — ignore */
  }
}

// ACA1 (session API) base URL, most-specific first:
//   1. runtime override (join-dialog "接続先サーバー")
//   2. build-time VITE_ACA1_URL (the deployed Azure ACA1)
//   3. this app's own origin (packaged Electron / a LAN browser on the host)
function aca1Base(): string {
  const runtime = getServerUrl();
  if (runtime) return runtime;
  const built = typeof __ACA1_URL__ === 'string' ? __ACA1_URL__ : '';
  if (built) return built.replace(/\/+$/, '');
  return window.location.origin;
}

// ACA1 records its reachable URL as PUBLIC_RELAY_URL; in local/LAN mode that
// is a loopback address. Swap the loopback host for the host we actually
// reached ACA1 on, keeping the relay's own port. A real Azure FQDN has no
// loopback host and is left untouched.
function rewriteLoopback(url: string): string {
  try {
    const u = new URL(url);
    if (u.hostname === 'localhost' || u.hostname === '127.0.0.1') {
      const target = new URL(aca1Base());
      u.protocol = target.protocol;
      u.hostname = target.hostname;
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

// The two uploaded YAML files are parsed and normalised *here*, with the same
// yamlService the app uses for File > Open, then sent to ACA1 as JSON — so the
// relay only ever stores a baseline the reducer/UI can consume directly.
export async function createSessionFromYaml(
  name: string, scheduleFile: File, envConfigFile: File,
): Promise<CreateResult> {
  const [scheduleText, envText] = await Promise.all([scheduleFile.text(), envConfigFile.text()]);
  let baseline: SessionBaseline;
  try {
    baseline = {
      schedule: parseScheduleYaml(scheduleText),
      envConfig: parseEnvConfigYaml(envText),
      currentView: 'worker',
    };
  } catch (err) {
    throw new Error(`YAML の解析に失敗しました: ${err instanceof Error ? err.message : String(err)}`);
  }
  return createSessionFromState(name, baseline);
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

// LAN IPs of the machine running this app's server (local/desktop mode only —
// 404s and returns [] on Azure). Shown to a host so they can tell teammates
// which address to enter as 接続先サーバー.
export async function fetchLanAddresses(): Promise<string[]> {
  try {
    const res = await fetch(`${aca1Base()}/api/network-info`);
    const data = await readJson(res);
    return Array.isArray(data.addresses) ? (data.addresses as string[]) : [];
  } catch {
    return [];
  }
}

/** The base URL the app is currently pointed at (own origin, or the override). */
export function currentServerBase(): string {
  return aca1Base();
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
