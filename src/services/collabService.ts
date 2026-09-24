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

// Set by probeAzureReachability() below — in-memory only (never persisted),
// separate from the user's own runtime override in localStorage. Starts
// false (optimistic: try the baked-in Azure URL) until the startup probe
// says otherwise.
let azureUnreachable = false;

// ACA1 (session API) base URL, most-specific first:
//   1. runtime override (join-dialog "接続先サーバー")
//   2. build-time VITE_ACA1_URL (the deployed Azure ACA1) — unless the
//      startup probe found it unreachable, in which case skip straight to (3)
//   3. this app's own origin (packaged Electron / a LAN browser on the host)
function aca1Base(): string {
  const runtime = getServerUrl();
  if (runtime) return runtime;
  const built = typeof __ACA1_URL__ === 'string' ? __ACA1_URL__ : '';
  if (built && !azureUnreachable) return built.replace(/\/+$/, '');
  return window.location.origin;
}

// Called once at app startup (AppContext) when a build-time Azure URL is
// baked in — a packaged installer built with .env.production's
// VITE_ACA1_URL should "just work" against the company's deployed backend,
// but must not get stuck trying to reach Azure with no network/VPN, so this
// does a quick, short-timeout reachability check up front and remembers the
// result for aca1Base() to use for the rest of the app's lifetime. A no-op
// if there's a runtime override already (the user/LAN-discovery picked an
// explicit server, which always wins) or no build-time URL at all.
//
// Memoized to a single in-flight/settled promise (idempotent — a second call
// just returns the first one) so every network call below can safely await
// it via ensureProbed() without re-probing. That await is what closes the
// startup race where a dialog opened in the first ~3s (before the probe
// resolves) would otherwise hit the unreachable Azure URL directly and show
// a scary "接続できません" error that only clears on the next 5s poll.
let probePromise: Promise<void> | null = null;

export function probeAzureReachability(): Promise<void> {
  if (probePromise) return probePromise;
  probePromise = (async () => {
    const built = typeof __ACA1_URL__ === 'string' ? __ACA1_URL__ : '';
    if (!built || getServerUrl()) return;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 3000);
      const res = await fetch(`${built.replace(/\/+$/, '')}/api/health`, { signal: controller.signal });
      clearTimeout(timer);
      azureUnreachable = !res.ok;
    } catch {
      azureUnreachable = true;
    }
  })();
  return probePromise;
}

async function ensureProbed(): Promise<void> {
  if (probePromise) await probePromise;
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
  await ensureProbed();
  const res = await fetch(`${aca1Base()}/api/sessions`);
  const data = await readJson(res);
  if (!res.ok || !data.ok || !Array.isArray(data.sessions)) {
    throw new Error((data.error as string) ?? 'セッション一覧の取得に失敗しました');
  }
  return data.sessions as SessionSummary[];
}

export async function createSessionFromState(name: string, baseline: SessionBaseline): Promise<CreateResult> {
  await ensureProbed();
  const res = await fetch(`${aca1Base()}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, ...baseline }),
  });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.sessionId) throw new Error((data.error as string) ?? 'セッションの作成に失敗しました');
  return { sessionId: data.sessionId as string, ownerToken: data.ownerToken as string };
}

// Parses two uploaded YAML files with the same yamlService the app uses for
// File > Open — the relay only ever stores/carries a baseline the reducer/UI
// can consume directly, never raw YAML text.
export async function parseYamlBaseline(scheduleFile: File, envConfigFile: File): Promise<SessionBaseline> {
  const [scheduleText, envText] = await Promise.all([scheduleFile.text(), envConfigFile.text()]);
  try {
    return {
      schedule: parseScheduleYaml(scheduleText),
      envConfig: parseEnvConfigYaml(envText),
      currentView: 'worker',
    };
  } catch (err) {
    throw new Error(`YAML の解析に失敗しました: ${err instanceof Error ? err.message : String(err)}`);
  }
}

export async function createSessionFromYaml(
  name: string, scheduleFile: File, envConfigFile: File,
): Promise<CreateResult> {
  const baseline = await parseYamlBaseline(scheduleFile, envConfigFile);
  return createSessionFromState(name, baseline);
}

// Overwrite an EXISTING session's whole data (same id) — used when the user
// confirms overwriting a duplicate-named session instead of creating a new
// one. Anyone currently connected to that session gets a live resync; no
// owner token needed (consistent with lock/unlock and the in-session update
// feature — this app doesn't gate collab actions on ownership).
export async function overwriteSessionState(sessionId: string, baseline: SessionBaseline): Promise<void> {
  await ensureProbed();
  const res = await fetch(`${aca1Base()}/api/sessions/${encodeURIComponent(sessionId)}/replace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(baseline),
  });
  const data = await readJson(res);
  if (!res.ok || !data.ok) throw new Error((data.error as string) ?? 'セッションの上書きに失敗しました');
}

export async function openSession(sessionId: string): Promise<{ relayUrl: string; status: SessionStatus }> {
  await ensureProbed();
  const res = await fetch(`${aca1Base()}/api/sessions/${encodeURIComponent(sessionId)}/open`, { method: 'POST' });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.relayUrl) throw new Error((data.error as string) ?? 'セッションを開けませんでした');
  return { relayUrl: rewriteLoopback(data.relayUrl as string), status: data.status as SessionStatus };
}

export async function deleteSession(sessionId: string, ownerToken: string): Promise<void> {
  await ensureProbed();
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
    await ensureProbed();
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
    await ensureProbed();
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

export interface LanHost {
  name: string;
  url: string;
  lastSeenAt: number;
}

// Other GanttChartEditor desktop apps the server at aca1Base() has heard on
// the LAN (zero-config discovery — server/src/lan/discoveryBeacon.ts). Uses
// the same target resolution as every other call (override → build URL →
// own origin) rather than always window.location.origin — that origin has no
// server behind it at all in dev/Azure (Vite's dev proxy would blindly
// forward to a fixed local port nothing is listening on, surfacing as a noisy
// ECONNREFUSED). ACA1/ACA2 answer this route with an always-empty list (LAN
// discovery isn't meaningful once there's a well-known server URL), so this
// never errors — it's just an empty result off the LAN/local role.
export async function fetchLanHosts(): Promise<LanHost[]> {
  try {
    await ensureProbed();
    const res = await fetch(`${aca1Base()}/api/lan-hosts`);
    const data = await readJson(res);
    return Array.isArray(data.hosts) ? (data.hosts as LanHost[]) : [];
  } catch {
    return [];
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

// Best-effort final snapshot sent by the last connected editor right before
// they leave (see AppContext.leaveCollabSession) — the server persists only
// ever a single "current state" snapshot per session (no action-by-action
// log), so this is how whatever was edited this session actually makes it to
// storage. Silently a no-op if the socket is already gone; nothing here is
// worth surfacing an error for on the way out the door.
export function sendCollabCheckpoint(baseline: SessionBaseline): void {
  socket?.emit('checkpoint', baseline);
}

// Explicit "replace this session's whole data" push — only takes effect
// while locked (server-enforced), open to any participant same as
// lock/unlock. The server broadcasts a fresh sync-init back to everyone
// (including the sender) once applied, which the existing onSyncInit
// handler already knows how to apply — no separate client-side event needed.
export function sendCollabSessionUpdate(baseline: SessionBaseline): void {
  socket?.emit('session-update', baseline);
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
