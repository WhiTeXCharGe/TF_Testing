import type { Server as HttpServer } from 'node:http';
import { Server, Socket } from 'socket.io';

// Fast, minimal "live view" broadcast: the host's app pushes its current
// schedule/envConfig whenever it changes, and every connected viewer socket
// gets it relayed live. No sessions, no auth, no persistence — a single
// in-memory "latest snapshot" per server process. Good enough for one host
// sharing their current file to read-only viewers on the same network.

interface ViewSnapshot {
  schedule: unknown;
  envConfig: unknown;
  currentView: 'worker' | 'device';
  updatedAt: number;
}

let latestSnapshot: ViewSnapshot | null = null;

export function createViewBroadcastServer(httpServer: HttpServer): Server {
  const io = new Server(httpServer, {
    path: '/live-view/socket.io',
    cors: { origin: true },
    maxHttpBufferSize: 20 * 1024 * 1024, // schedules can be a few MB of JSON
  });

  io.on('connection', (socket: Socket) => {
    if (latestSnapshot) {
      socket.emit('view-state', latestSnapshot);
    }

    socket.on('host-update', (payload: { schedule: unknown; envConfig: unknown; currentView: 'worker' | 'device' }) => {
      latestSnapshot = { ...payload, updatedAt: Date.now() };
      socket.broadcast.emit('view-state', latestSnapshot);
    });

    socket.on('host-stop', () => {
      latestSnapshot = null;
      socket.broadcast.emit('view-stopped');
    });
  });

  return io;
}
