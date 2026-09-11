// Pure in-memory "who have we heard broadcasting" store — no sockets here, so
// it's unit-testable directly. discoveryBeacon.ts feeds it from UDP messages.

export interface LanHost {
  name: string;
  url: string;
  lastSeenAt: number;
}

export interface PeerRegistry {
  /** Record/refresh a sighting of a peer. */
  see(host: { name: string; url: string }, now?: number): void;
  /** Drop peers not seen within the registry's staleness window. */
  prune(now?: number): void;
  /** Every currently-known (non-stale) peer. */
  list(): LanHost[];
}

export function createPeerRegistry(staleAfterMs: number): PeerRegistry {
  const peers = new Map<string, LanHost>();

  return {
    see(host, now = Date.now()) {
      peers.set(host.url, { name: host.name, url: host.url, lastSeenAt: now });
    },
    prune(now = Date.now()) {
      for (const [url, host] of peers) {
        if (now - host.lastSeenAt > staleAfterMs) peers.delete(url);
      }
    },
    list() {
      return [...peers.values()];
    },
  };
}
