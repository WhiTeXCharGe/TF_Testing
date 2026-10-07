import { io, Socket } from 'socket.io-client';
import type {
  SessionBaseline, SessionParticipant, SessionRole, SessionConnectionStatus,
  SessionStatus, SessionSummary,
} from '../types/appState';
import { parseScheduleYaml, parseEnvConfigYaml } from './yamlService';
import { loadAppConfig, getLoadedAppConfig, isOnlineMode } from './appConfig';
import { ServerUnreachableError, ConfigError } from './serverErrors';

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
  /**
   * The server is asking every connected editor in this session to send a
   * fresh checkpoint (see requestAllCheckpoints server-side — used before a
   * replica shuts down, e.g. Azure scaling ACA2 to zero) so a session's
   * durable snapshot doesn't go stale just because nobody happened to be the
   * "last one out" at that exact moment.
   */
  onCheckpointRequest: () => void;
}

// Runtime override: the desktop app / LAN host address a participant types in
// the join dialog. Blank = talk to this app's own bundled server (the
// packaged Electron window loads it from the same origin).
const SERVER_URL_KEY = 'gantt.collab.serverUrl';

export function getServerUrl(): string {
  // config.txt mode=online pins the app to the Azure server — a remembered
  // LAN host must never leak back in.
  if (isOnlineMode()) return '';
  try {
    return localStorage.getItem(SERVER_URL_KEY) ?? '';
  } catch {
    return '';
  }
}

export function setServerUrl(url: string): void {
  if (isOnlineMode()) return;
  try {
    const trimmed = url.trim().replace(/\/+$/, '');
    if (trimmed) localStorage.setItem(SERVER_URL_KEY, trimmed);
    else localStorage.removeItem(SERVER_URL_KEY);
  } catch {
    /* storage disabled — ignore */
  }
}

// Azure Container Apps can scale to zero; a cold start takes a while.
const ONLINE_REQUEST_TIMEOUT_MS = 60_000;

async function ensureConfig(): Promise<void> {
  const config = await loadAppConfig();
  if (config.mode === 'error') throw new ConfigError(config.message);
}

// ACA1 (session API) base URL:
//   - config.txt mode=online → the configured Azure URL, and nothing else:
//     no runtime override, no fallback to this PC if it can't be reached.
//   - otherwise (mode=local / dev) → runtime override (a LAN host found by
//     discovery), else this app's own origin (packaged Electron / a LAN
//     browser on the host).
function aca1Base(): string {
  const config = getLoadedAppConfig();
  if (config?.mode === 'online') return config.azureUrl;
  const runtime = getServerUrl();
  if (runtime) return runtime;
  return window.location.origin;
}

// One place every ACA1 call goes through. Online mode turns "couldn't get an
// answer" into a ServerUnreachableError so the UI can say so (instead of
// showing it as an empty session list or a vague failure).
async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  await ensureConfig();
  const base = aca1Base();
  const url = `${base}${path}`;
  if (!isOnlineMode()) return init ? fetch(url, init) : fetch(url);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ONLINE_REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (res.status === 502 || res.status === 503 || res.status === 504) throw new ServerUnreachableError(base);
    return res;
  } catch (err) {
    if (err instanceof ServerUnreachableError) throw err;
    throw new ServerUnreachableError(base);
  } finally {
    clearTimeout(timer);
  }
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
  const res = await apiFetch('/api/sessions');
  const data = await readJson(res);
  if (!res.ok || !data.ok || !Array.isArray(data.sessions)) {
    throw new Error((data.error as string) ?? 'セッション一覧の取得に失敗しました');
  }
  return data.sessions as SessionSummary[];
}

export async function createSessionFromState(name: string, baseline: SessionBaseline): Promise<CreateResult> {
  const res = await apiFetch('/api/sessions', {
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
  const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/replace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(baseline),
  });
  const data = await readJson(res);
  if (!res.ok || !data.ok) throw new Error((data.error as string) ?? 'セッションの上書きに失敗しました');
}

export async function openSession(sessionId: string): Promise<{ relayUrl: string; status: SessionStatus }> {
  const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/open`, { method: 'POST' });
  const data = await readJson(res);
  if (!res.ok || !data.ok || !data.relayUrl) throw new Error((data.error as string) ?? 'セッションを開けませんでした');
  return { relayUrl: rewriteLoopback(data.relayUrl as string), status: data.status as SessionStatus };
}

// No owner-token gate server-side (see sessionApi.ts's DELETE handler) — a
// shared admin action reachable from 編集 > オンラインセッションを削除 for
// any session in the list, not just ones this client created.
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!res.ok) {
    const data = await readJson(res);
    throw new Error((data.error as string) ?? 'セッションの削除に失敗しました');
  }
}

export async function fetchSessionName(sessionId: string): Promise<string | null> {
  try {
    const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
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
    await ensureConfig();
    if (isOnlineMode()) return []; // no LAN to advertise when pinned to Azure
    const res = await apiFetch('/api/network-info');
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
    await ensureConfig();
    if (isOnlineMode()) return []; // LAN discovery is meaningless when pinned to Azure
    const res = await apiFetch('/api/lan-hosts');
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
  const handleCheckpointRequest = () => cb.onCheckpointRequest();

  s.on('connect', handleConnect);
  s.on('sync-init', handleSyncInit);
  s.on('action', handleAction);
  s.on('presence', handlePresence);
  s.on('session-status', handleSessionStatus);
  s.on('disconnect', handleDisconnect);
  s.on('checkpoint-request', handleCheckpointRequest);

  if (s.connected) handleConnect();

  return () => {
    s.off('connect', handleConnect);
    s.off('sync-init', handleSyncInit);
    s.off('action', handleAction);
    s.off('presence', handlePresence);
    s.off('session-status', handleSessionStatus);
    s.off('disconnect', handleDisconnect);
    s.off('checkpoint-request', handleCheckpointRequest);
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

// A full-state snapshot the client hands the server: on leaving as the last
// editor (see AppContext.leaveCollabSession), in response to a server-side
// checkpoint-request (see onCheckpointRequest above), on a periodic backup
// timer, or right before an Electron window closes. The server persists only
// ever a single "current state" snapshot per session (no action-by-action
// log), so this is the only way whatever was edited this session actually
// makes it to storage. Resolves once the server has actually written it (or
// false if there's no socket, the server rejected it, or it didn't ack
// within the timeout) so a caller that needs to know the write landed before
// proceeding — e.g. an Electron close-intercept — can await it.
const CHECKPOINT_ACK_TIMEOUT_MS = 5000;

export function sendCollabCheckpoint(baseline: SessionBaseline): Promise<boolean> {
  return new Promise((resolve) => {
    if (!socket) { resolve(false); return; }
    let settled = false;
    const timer = setTimeout(() => { if (!settled) { settled = true; resolve(false); } }, CHECKPOINT_ACK_TIMEOUT_MS);
    socket.emit('checkpoint', baseline, (ok: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(!!ok);
    });
  });
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