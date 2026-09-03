import { describe, it, expect, beforeEach } from 'vitest';
import {
  createSession, getSession, getSessionName, appendAction, addParticipant, removeParticipant,
  sweepIdleSessions, _resetForTests,
} from './sessionStore.js';

const BASELINE = { schedule: { foo: 'bar' }, envConfig: { baz: 1 }, currentView: 'worker' as const };

beforeEach(() => _resetForTests());

describe('createSession / getSession', () => {
  it('creates a session with the given baseline and no actions or participants', () => {
    const id = createSession('Test Session', BASELINE);
    const session = getSession(id);
    expect(session).not.toBeNull();
    expect(session?.baseline).toEqual(BASELINE);
    expect(session?.actions).toEqual([]);
    expect(session?.participants).toEqual([]);
  });

  it('returns null for an unknown session id', () => {
    expect(getSession('does-not-exist')).toBeNull();
  });

  it('returns defensive copies — mutations to returned data do not affect internal state', () => {
    const id = createSession('Test Session', BASELINE);
    appendAction(id, 'SET_SCHEDULE', { hello: 'world' });

    // Get the session and attempt to mutate the returned data
    const session1 = getSession(id)!;
    (session1.actions as any).push({ seq: 999, type: 'FAKE_ACTION', payload: {} });
    (session1.baseline as any).schedule = { mutated: true };

    // Fetch again and verify internal state is unaffected
    const session2 = getSession(id)!;
    expect(session2.actions).toHaveLength(1);
    expect(session2.actions[0]).toEqual({ seq: 0, type: 'SET_SCHEDULE', payload: { hello: 'world' } });
    expect(session2.baseline).toEqual(BASELINE);
  });

  describe('createSession / getSession — name', () => {
    it('stores and returns the session name', () => {
      const id = createSession('My Session', BASELINE);
      expect(getSession(id)?.name).toBe('My Session');
    });
  });
});

describe('appendAction', () => {
  it('assigns increasing sequence numbers and stores the action', () => {
    const id = createSession('Test Session', BASELINE);
    const a1 = appendAction(id, 'SET_SCHEDULE', { hello: 'world' });
    const a2 = appendAction(id, 'UPDATE_PLAN_RANGE', { startDate: '2026-01-01', endDate: '2026-01-31' });
    expect(a1).toEqual({ seq: 0, type: 'SET_SCHEDULE', payload: { hello: 'world' } });
    expect(a2).toEqual({ seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-01-01', endDate: '2026-01-31' } });
    expect(getSession(id)?.actions).toEqual([a1, a2]);
  });

  it('returns null for an unknown session id', () => {
    expect(appendAction('does-not-exist', 'SET_SCHEDULE', {})).toBeNull();
  });
});

describe('addParticipant / removeParticipant', () => {
  it('adds a participant and returns the full list', () => {
    const id = createSession('Test Session', BASELINE);
    const list = addParticipant(id, 'p1', 'Alice', 'edit');
    expect(list).toEqual([{ id: 'p1', name: 'Alice', role: 'edit' }]);
  });

  it('removes a participant and returns the remaining list', () => {
    const id = createSession('Test Session', BASELINE);
    addParticipant(id, 'p1', 'Alice', 'edit');
    addParticipant(id, 'p2', 'Bob', 'view');
    const list = removeParticipant(id, 'p1');
    expect(list).toEqual([{ id: 'p2', name: 'Bob', role: 'view' }]);
  });

  it('returns null for an unknown session id', () => {
    expect(addParticipant('does-not-exist', 'p1', 'Alice', 'edit')).toBeNull();
    expect(removeParticipant('does-not-exist', 'p1')).toBeNull();
  });
});

describe('sweepIdleSessions', () => {
  it('removes sessions with zero participants past the idle threshold', () => {
    const id = createSession('Test Session', BASELINE);
    const removed = sweepIdleSessions(1000, Date.now() + 2000);
    expect(removed).toBe(1);
    expect(getSession(id)).toBeNull();
  });

  it('keeps sessions that still have participants', () => {
    const id = createSession('Test Session', BASELINE);
    addParticipant(id, 'p1', 'Alice', 'edit');
    const removed = sweepIdleSessions(1000, Date.now() + 2000);
    expect(removed).toBe(0);
    expect(getSession(id)).not.toBeNull();
  });

  it('keeps sessions inside the idle threshold', () => {
    const id = createSession('Test Session', BASELINE);
    const removed = sweepIdleSessions(60_000, Date.now() + 1000);
    expect(removed).toBe(0);
    expect(getSession(id)).not.toBeNull();
  });
});

describe('getSessionName', () => {
  it('returns the name for a real session', () => {
    const id = createSession('Weekly Plan', BASELINE);
    expect(getSessionName(id)).toBe('Weekly Plan');
  });

  it('returns null for an unknown session id', () => {
    expect(getSessionName('does-not-exist')).toBeNull();
  });
});
