import { describe, it, expect } from 'vitest';
import type { SessionStatus, SessionBaseline, LoggedAction } from './types.js';

describe('shared collab types', () => {
  it('SessionStatus is one of the three literals', () => {
    const values: SessionStatus[] = ['open', 'lock', 'close'];
    expect(values).toHaveLength(3);
  });

  it('a baseline and an action are structurally usable', () => {
    const baseline: SessionBaseline = { schedule: {}, envConfig: {}, currentView: 'worker' };
    const action: LoggedAction = { seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } };
    expect(baseline.currentView).toBe('worker');
    expect(action.seq).toBe(0);
  });
});
