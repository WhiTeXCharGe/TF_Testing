import dgram from 'node:dgram';
import { createPeerRegistry, type LanHost } from './peerRegistry.js';

// Zero-config LAN discovery for ROLE=local (the desktop app): every instance
// periodically UDP-broadcasts "here I am" and listens for the same broadcast
// from other instances, so a participant can pick a host from a list instead
// of typing its IP. Best-effort only — a firewalled/segmented network just
// means nothing is discovered and the manual 接続先サーバー field still works;
// this never throws or crashes the app.

const DISCOVERY_PORT = 41237;
const BROADCAST_ADDR = '255.255.255.255';
const BROADCAST_INTERVAL_MS = 2000;
const STALE_AFTER_MS = 8000;
const MAGIC = 'gantt-collab-local-v1';

interface Beacon {
  magic: string;
  name: string;
  url: string;
}

export interface DiscoveryBeacon {
  getKnownHosts(): LanHost[];
  stop(): void;
}

export function startDiscoveryBeacon(selfUrl: string, selfName: string): DiscoveryBeacon {
  const registry = createPeerRegistry(STALE_AFTER_MS);
  const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
  let broadcastTimer: NodeJS.Timeout | undefined;
  let pruneTimer: NodeJS.Timeout | undefined;

  socket.on('error', () => {
    // Best-effort feature: swallow bind/send errors (port in use, no
    // broadcast-capable interface, sandboxed network, ...) rather than
    // taking the whole app down over session-discovery convenience.
  });

  socket.on('message', (msg) => {
    try {
      const data = JSON.parse(msg.toString('utf-8')) as Partial<Beacon>;
      if (data.magic !== MAGIC || typeof data.name !== 'string' || typeof data.url !== 'string') return;
      if (data.url === selfUrl) return; // don't list ourselves
      registry.see({ name: data.name, url: data.url });
    } catch {
      /* not our beacon format — ignore */
    }
  });

  try {
    socket.bind(DISCOVERY_PORT, () => {
      try {
        socket.setBroadcast(true);
      } catch {
        return;
      }
      const payload = Buffer.from(JSON.stringify({ magic: MAGIC, name: selfName, url: selfUrl } satisfies Beacon));
      const broadcast = () => socket.send(payload, DISCOVERY_PORT, BROADCAST_ADDR, () => {});
      broadcast();
      broadcastTimer = setInterval(broadcast, BROADCAST_INTERVAL_MS);
      broadcastTimer.unref();
    });
  } catch {
    /* bind can throw synchronously on some platforms — ignore, feature stays off */
  }

  pruneTimer = setInterval(() => registry.prune(), BROADCAST_INTERVAL_MS);
  pruneTimer.unref();

  return {
    getKnownHosts: () => registry.list(),
    stop: () => {
      clearInterval(broadcastTimer);
      clearInterval(pruneTimer);
      try { socket.close(); } catch { /* already closed */ }
    },
  };
}
