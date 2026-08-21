import { io, Socket } from 'socket.io-client';
import type { ScheduleData } from '../types/schedule';
import type { EnvConfig } from '../types/envConfig';
import type { ViewMode } from '../types/appState';

export interface ViewSnapshot {
  schedule: ScheduleData;
  envConfig: EnvConfig;
  currentView: ViewMode;
  updatedAt: number;
}

export type ViewConnectionStatus = 'disconnected' | 'connecting' | 'connected';

// Fast, minimal "live view" broadcast: one host pushes its current file,
// any number of read-only viewers receive it live. No sessions, no auth —
// see server/src/collab/viewBroadcast.ts for the matching server side.

// The API/socket server always runs on port 3010 on the same machine as the
// frontend, whether reached as localhost (host) or a LAN IP (viewer who
// opened the shared link) — window.location.hostname already reflects
// whichever address was actually used to load the page.
function getSocketOrigin(): string {
  return `${window.location.protocol}//${window.location.hostname}:3010`;
}

let socket: Socket | null = null;

function ensureSocket(): Socket {
  if (!socket) {
    socket = io(getSocketOrigin(), {
      path: '/live-view/socket.io',
      transports: ['websocket', 'polling'],
    });
  }
  return socket;
}

export function startHostBroadcast(): void {
  ensureSocket();
}

export function sendHostUpdate(schedule: ScheduleData, envConfig: EnvConfig, currentView: ViewMode): void {
  ensureSocket().emit('host-update', { schedule, envConfig, currentView });
}

export function stopHostBroadcast(): void {
  socket?.emit('host-stop');
  socket?.disconnect();
  socket = null;
}

export function connectAsViewer(
  onState: (snapshot: ViewSnapshot) => void,
  onStatusChange: (status: ViewConnectionStatus) => void,
): () => void {
  const s = ensureSocket();
  onStatusChange('connecting');

  const handleConnect = () => onStatusChange('connected');
  const handleDisconnect = () => onStatusChange('disconnected');
  const handleViewState = (snapshot: ViewSnapshot) => onState(snapshot);
  const handleViewStopped = () => onStatusChange('disconnected');

  s.on('connect', handleConnect);
  s.on('disconnect', handleDisconnect);
  s.on('view-state', handleViewState);
  s.on('view-stopped', handleViewStopped);

  return () => {
    s.off('connect', handleConnect);
    s.off('disconnect', handleDisconnect);
    s.off('view-state', handleViewState);
    s.off('view-stopped', handleViewStopped);
    s.disconnect();
    socket = null;
  };
}

// Builds a link a viewer on another PC can open directly, e.g.
// http://192.168.1.23:5173/?view=1 — reuses whatever port/path the host is
// currently on (works the same in dev, where the frontend and API are on
// different ports, and in a packaged build, where they're the same origin).
export async function fetchShareableLink(): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/network-info`);
  const data = (await res.json()) as { ok: boolean; addresses: string[] };
  const lanIp = data.addresses[0] ?? window.location.hostname;
  return `${window.location.protocol}//${lanIp}:${window.location.port}${window.location.pathname}?view=1`;
}
