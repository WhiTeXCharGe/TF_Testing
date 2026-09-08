import { describe, it, expect, beforeEach } from 'vitest';
import { createMemStorage } from './storage/memStorage.js';
import type { StorageClient } from './storage/storageClient.js';
import { createSessionStore, type SessionStore } from './sessionStore.js';
import { createSessionRecord, hashOwnerToken, loadSessionRecord } from './persistence.js';

const BASELINE = { schedule: { foo: 'bar' }, envConfig: { baz: 1 }, currentView: 'worker' as const };

let storage: StorageClient;
let store: SessionStore;
let id: string;

beforeEach(async () => {
  storage = createMemStorage();
  store = createSessionStore({ storage });
  id = await createSessionRecord(storage, {
    name: 'Weekly Plan', baseline: BASELINE, ownerTokenHash: hashOwnerToken('owner-secret'),
  });
});

describe('activateFromStorage', () => {
  it('loads a stored record into memory', async () => {
    expect(store.isLoaded(id)).toBe(false);
    expect(store.getSession(id)).toBeNull();
    expect(await store.activateFromStorage(id)).toBe(true);
    expect(store.isLoaded(id)).toBe(true);
    expect(store.getSession(id)).toEqual({
      name: 'Weekly Plan', baseline: BASELINE, actions: [], participants: [], status: 'open',
    });
  });

  it('returns false for an unknown id', async () => {
    expect(await store.activateFromStorage('does-not-exist')).toBe(false);
  });

  it('re-activating a loaded session is a no-op returning true', async () => {
    await store.activateFromStorage(id);
    store.appendAction(id, 'SET_SCHEDULE', { a: 1 });
    expect(await store.activateFromStorage(id)).toBe(true);
    expect(store.getSession(id)?.actions).toHaveLength(1); // not reloaded/reset
  });
});

describe('appendAction', () => {
  it('assigns increasing seq and marks the session dirty', async () => {
    await store.activateFromStorage(id);
    expect(store.appendAction(id, 'SET_SCHEDULE', { a: 1 })).toEqual({ seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } });
    expect(store.appendAction(id, 'UPDATE_PLAN_RANGE', { s: 'x' })).toEqual({ seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { s: 'x' } });
  });

  it('returns null for a session that is not loaded', () => {
    expect(store.appendAction(id, 'SET_SCHEDULE', {})).toBeNull();
  });
});

describe('flush / re-activation', () => {
  it('flush writes the log so a fresh store re-activates with the actions', async () => {
    await store.activateFromStorage(id);
    store.appendAction(id, 'SET_SCHEDULE', { a: 1 });
    await store.flush(id);

    const fresh = createSessionStore({ storage });
    await fresh.activateFromStorage(id);
    expect(fresh.getSession(id)?.actions).toEqual([{ seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } }]);
    // nextSeq continues after the reloaded log
    expect(fresh.appendAction(id, 'X', {})?.seq).toBe(1);
  });
});

describe('participants', () => {
  it('add / remove / count', async () => {
    await store.activateFromStorage(id);
    expect(store.addParticipant(id, 'p1', 'Alice', 'edit')).toEqual([{ id: 'p1', name: 'Alice', role: 'edit' }]);
    store.addParticipant(id, 'p2', 'Bob', 'view');
    expect(store.participantCount(id)).toBe(2);
    expect(store.removeParticipant(id, 'p1')).toEqual([{ id: 'p2', name: 'Bob', role: 'view' }]);
    expect(store.participantCount(id)).toBe(1);
  });
});

describe('lock', () => {
  it('setLocked flips status; getSession reflects it', async () => {
    await store.activateFromStorage(id);
    expect(store.setLocked(id, true)).toBe('lock');
    expect(store.getSession(id)?.status).toBe('lock');
    expect(store.setLocked(id, false)).toBe('open');
    expect(store.getSession(id)?.status).toBe('open');
  });

  it('a session activated from a locked record comes back locked', async () => {
    await store.activateFromStorage(id);
    store.setLocked(id, true);
    await store.flush(id);
    const fresh = createSessionStore({ storage });
    await fresh.activateFromStorage(id);
    expect(fresh.getSession(id)?.status).toBe('lock');
  });
});

describe('evict / markClosed', () => {
  it('evict flushes then unloads', async () => {
    await store.activateFromStorage(id);
    store.appendAction(id, 'SET_SCHEDULE', { a: 1 });
    await store.evict(id);
    expect(store.isLoaded(id)).toBe(false);
    expect((await loadSessionRecord(storage, id))?.log).toHaveLength(1);
  });

  it('markClosed sets the at-rest status to close with no relay pointer', async () => {
    await store.markClosed(id);
    const rec = await loadSessionRecord(storage, id);
    expect(rec?.status).toMatchObject({ status: 'close', relayInstance: null, relayUrl: null });
  });
});

describe('markActivated', () => {
  it('records the replica and sets status open', async () => {
    await store.activateFromStorage(id);
    const status = await store.markActivated(id, { relayInstance: 'replica-7', relayUrl: 'http://relay:4010' });
    expect(status).toBe('open');
    const rec = await loadSessionRecord(storage, id);
    expect(rec?.status).toMatchObject({ status: 'open', relayInstance: 'replica-7', relayUrl: 'http://relay:4010' });
  });
});

describe('getLive', () => {
  it('reports active + count + status for a loaded session', async () => {
    await store.activateFromStorage(id);
    store.addParticipant(id, 'p1', 'Alice', 'edit');
    expect(await store.getLive(id)).toEqual({ active: true, participantCount: 1, status: 'open' });
  });

  it('reads status from storage when the session is not loaded here', async () => {
    expect(await store.getLive(id)).toEqual({ active: false, participantCount: 0, status: 'close' });
  });
});

describe('ownerTokenHash', () => {
  it('returns the seeded hash once loaded, null before', async () => {
    expect(store.ownerTokenHash(id)).toBeNull();
    await store.activateFromStorage(id);
    expect(store.ownerTokenHash(id)).toBe(hashOwnerToken('owner-secret'));
  });
});

describe('sweepIdleSessions', () => {
  it('drops loaded sessions with no participants past the idle window', async () => {
    await store.activateFromStorage(id);
    expect(store.sweepIdleSessions(1000, Date.now() + 2000)).toBe(1);
    expect(store.isLoaded(id)).toBe(false);
  });

  it('keeps sessions that still have participants', async () => {
    await store.activateFromStorage(id);
    store.addParticipant(id, 'p1', 'Alice', 'edit');
    expect(store.sweepIdleSessions(1000, Date.now() + 2000)).toBe(0);
    expect(store.isLoaded(id)).toBe(true);
  });
});
