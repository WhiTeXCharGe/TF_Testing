import { describe, it, expect } from 'vitest';
import { createMemStorage } from './memStorage.js';

describe('memStorage', () => {
  it('round-trips JSON and returns null for missing keys', async () => {
    const s = createMemStorage();
    expect(await s.getJson('a')).toBeNull();
    await s.putJson('a', { v: 1 });
    expect(await s.getJson('a')).toEqual({ v: 1 });
  });

  it('lists and deletes by prefix', async () => {
    const s = createMemStorage();
    await s.putJson('sessions/a/meta.json', {});
    await s.putJson('sessions/a/log.json', []);
    await s.putJson('sessions/b/meta.json', {});
    expect(await s.listPrefix('sessions/a')).toEqual(['sessions/a/log.json', 'sessions/a/meta.json']);
    await s.deletePrefix('sessions/a');
    expect(await s.listPrefix('sessions/')).toEqual(['sessions/b/meta.json']);
  });

  it('does not share state between instances', async () => {
    const a = createMemStorage();
    const b = createMemStorage();
    await a.putJson('k', { from: 'a' });
    expect(await b.getJson('k')).toBeNull();
  });
});
