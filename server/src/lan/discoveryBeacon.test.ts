import { describe, it, expect, afterEach } from 'vitest';
import { startDiscoveryBeacon, type DiscoveryBeacon } from './discoveryBeacon.js';

// Real UDP broadcast on the loopback-capable interface — exercises the
// actual wire format, not just the pure registry (see peerRegistry.test.ts).
// Best-effort by design (see discoveryBeacon.ts), so this polls rather than
// asserting on a fixed delay.

const beacons: DiscoveryBeacon[] = [];
afterEach(() => {
  for (const b of beacons.splice(0)) b.stop();
});

async function waitFor<T>(fn: () => T, predicate: (v: T) => boolean, timeoutMs = 6000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  let last: T;
  do {
    last = fn();
    if (predicate(last)) return last;
    await new Promise((r) => setTimeout(r, 100));
  } while (Date.now() < deadline);
  return last;
}

describe('startDiscoveryBeacon', () => {
  it('two beacons on the same LAN discover each other by name', async () => {
    const a = startDiscoveryBeacon('http://192.168.99.5:3010', 'PC-A');
    const b = startDiscoveryBeacon('http://192.168.99.6:3010', 'PC-B');
    beacons.push(a, b);

    const aSees = await waitFor(() => a.getKnownHosts(), (list) => list.length > 0);
    const bSees = await waitFor(() => b.getKnownHosts(), (list) => list.length > 0);

    expect(aSees).toEqual([{ name: 'PC-B', url: 'http://192.168.99.6:3010', lastSeenAt: expect.any(Number) }]);
    expect(bSees).toEqual([{ name: 'PC-A', url: 'http://192.168.99.5:3010', lastSeenAt: expect.any(Number) }]);
  }, 10_000);

  it('never lists itself', async () => {
    const a = startDiscoveryBeacon('http://192.168.99.7:3010', 'Solo');
    beacons.push(a);
    await new Promise((r) => setTimeout(r, 500));
    expect(a.getKnownHosts()).toEqual([]);
  });
});
