import { describe, it, expect } from 'vitest';
import { createPeerRegistry } from './peerRegistry.js';

describe('createPeerRegistry', () => {
  it('starts empty', () => {
    expect(createPeerRegistry(8000).list()).toEqual([]);
  });

  it('records a sighting and lists it', () => {
    const reg = createPeerRegistry(8000);
    reg.see({ name: 'DESKTOP-A', url: 'http://192.168.1.5:3010' }, 1000);
    expect(reg.list()).toEqual([{ name: 'DESKTOP-A', url: 'http://192.168.1.5:3010', lastSeenAt: 1000 }]);
  });

  it('a later sighting of the same url updates it in place, not duplicates it', () => {
    const reg = createPeerRegistry(8000);
    reg.see({ name: 'DESKTOP-A', url: 'http://192.168.1.5:3010' }, 1000);
    reg.see({ name: 'DESKTOP-A', url: 'http://192.168.1.5:3010' }, 2000);
    expect(reg.list()).toEqual([{ name: 'DESKTOP-A', url: 'http://192.168.1.5:3010', lastSeenAt: 2000 }]);
  });

  it('tracks multiple distinct peers', () => {
    const reg = createPeerRegistry(8000);
    reg.see({ name: 'A', url: 'http://192.168.1.5:3010' }, 1000);
    reg.see({ name: 'B', url: 'http://192.168.1.6:3010' }, 1000);
    expect(reg.list().map((h) => h.name).sort()).toEqual(['A', 'B']);
  });

  it('prune drops peers not seen within the staleness window', () => {
    const reg = createPeerRegistry(8000);
    reg.see({ name: 'A', url: 'http://192.168.1.5:3010' }, 1000);
    reg.prune(1000 + 8000); // exactly at the edge — still fresh (not > window)
    expect(reg.list()).toHaveLength(1);
    reg.prune(1000 + 8001); // now stale
    expect(reg.list()).toEqual([]);
  });

  it('prune keeps peers refreshed by a later sighting', () => {
    const reg = createPeerRegistry(8000);
    reg.see({ name: 'A', url: 'http://192.168.1.5:3010' }, 1000);
    reg.see({ name: 'A', url: 'http://192.168.1.5:3010' }, 6000);
    reg.prune(9000); // 3s since the refresh — still within the 8s window
    expect(reg.list()).toHaveLength(1);
  });
});
