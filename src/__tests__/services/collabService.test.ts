/**
 * @jest-environment jsdom
 */
import { io } from 'socket.io-client';
import {
  joinCollabRoom, fetchSessionName, openSession, listSessions, createSessionFromYaml,
  getServerUrl, setServerUrl,
} from '../../services/collabService';
import { SessionBaseline } from '../../types/appState';

jest.mock('socket.io-client', () => ({ io: jest.fn() }));

type Handler = (payload: never) => void;

const BASELINE: SessionBaseline = {
  schedule: { planRange: { startDate: '2026-01-01', endDate: '2026-01-31' }, workflowTaskList: [], assignmentList: [] },
  envConfig: { workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [] },
  currentView: 'worker',
};

function makeFakeSocket() {
  const handlers: Record<string, Handler[]> = {};
  return {
    connected: false,
    emit: jest.fn(),
    disconnect: jest.fn(),
    on(ev: string, fn: Handler) { (handlers[ev] ??= []).push(fn); },
    off(ev: string, fn: Handler) { handlers[ev] = (handlers[ev] ?? []).filter(h => h !== fn); },
    fire(ev: string, payload: unknown) { for (const h of [...(handlers[ev] ?? [])]) h(payload as never); },
  };
}

let fakeSocket: ReturnType<typeof makeFakeSocket>;
const openJoins: (() => void)[] = [];

beforeEach(() => {
  jest.clearAllMocks();
  fakeSocket = makeFakeSocket();
  (io as jest.Mock).mockReturnValue(fakeSocket);
});

afterEach(() => {
  for (const d of openJoins.splice(0)) d();
});

function join(isCreator: boolean) {
  const cb = {
    onSyncInit: jest.fn(),
    onAction: jest.fn(),
    onPresence: jest.fn(),
    onStatusChange: jest.fn(),
    onSessionStatus: jest.fn(),
  };
  const disconnect = joinCollabRoom('s1', 'Alice', 'edit', isCreator, 'http://relay:4010', undefined, cb);
  openJoins.push(disconnect);
  return cb;
}

const SYNC_INIT_OK = { ok: true, name: 'Test Session', baseline: BASELINE, actions: [], participants: [], status: 'open' as const };

it('connects the socket to the given relay URL', () => {
  join(false);
  expect(io).toHaveBeenCalledWith('http://relay:4010', expect.objectContaining({ path: '/collab/socket.io' }));
});

it('skips the baseline replay for the creator on their first sync-init', () => {
  const cb = join(true);
  fakeSocket.fire('sync-init', SYNC_INIT_OK);
  expect(cb.onSyncInit).not.toHaveBeenCalled();
  expect(cb.onSessionStatus).toHaveBeenCalledWith('open');
});

it('replays the baseline for the creator on a later (reconnect) sync-init', () => {
  const cb = join(true);
  fakeSocket.fire('sync-init', SYNC_INIT_OK);
  fakeSocket.fire('sync-init', {
    ...SYNC_INIT_OK,
    actions: [{ seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-05-01', endDate: '2026-05-31' } }],
  });
  expect(cb.onSyncInit).toHaveBeenCalledTimes(1);
  expect(cb.onSyncInit).toHaveBeenCalledWith('Test Session', BASELINE, [
    { seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-05-01', endDate: '2026-05-31' } },
  ]);
});

it('replays the baseline for a non-creator on every sync-init', () => {
  const cb = join(false);
  fakeSocket.fire('sync-init', SYNC_INIT_OK);
  fakeSocket.fire('sync-init', SYNC_INIT_OK);
  expect(cb.onSyncInit).toHaveBeenCalledTimes(2);
});

it('reports disconnected and replays nothing when sync-init comes back not-ok', () => {
  const cb = join(false);
  fakeSocket.fire('sync-init', { ok: false });
  expect(cb.onSyncInit).not.toHaveBeenCalled();
  expect(cb.onStatusChange).toHaveBeenLastCalledWith('disconnected');
});

it('forwards a session-status event to onSessionStatus', () => {
  const cb = join(false);
  fakeSocket.fire('session-status', { status: 'lock' });
  expect(cb.onSessionStatus).toHaveBeenCalledWith('lock');
});

it('emits ownerToken in the join payload when given', () => {
  const cb = {
    onSyncInit: jest.fn(), onAction: jest.fn(), onPresence: jest.fn(),
    onStatusChange: jest.fn(), onSessionStatus: jest.fn(),
  };
  const d = joinCollabRoom('s1', 'Alice', 'edit', true, 'http://relay:4010', 'secret-token', cb);
  openJoins.push(d);
  fakeSocket.fire('connect', undefined);
  expect(fakeSocket.emit).toHaveBeenCalledWith('join', { sessionId: 's1', name: 'Alice', role: 'edit', ownerToken: 'secret-token' });
});

it('openSession rewrites a loopback relay host to the current page host', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true, json: async () => ({ ok: true, relayUrl: 'http://localhost:4010', status: 'open' }),
  }) as never;
  const res = await openSession('abc');
  expect(res.status).toBe('open');
  expect(res.relayUrl).toBe(`${window.location.protocol}//${window.location.hostname}:4010`);
});

it('listSessions returns the sessions array', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true, json: async () => ({ ok: true, sessions: [{ id: 's1', name: 'A', status: 'open', participantCount: 2 }] }),
  }) as never;
  const list = await listSessions();
  expect(list).toEqual([{ id: 's1', name: 'A', status: 'open', participantCount: 2 }]);
});

describe('server URL override (desktop / LAN host)', () => {
  afterEach(() => setServerUrl(''));

  it('get/set round-trips through localStorage; blank clears it', () => {
    expect(getServerUrl()).toBe('');
    setServerUrl('http://192.168.1.9:3010/');
    expect(getServerUrl()).toBe('http://192.168.1.9:3010'); // trailing slash trimmed
    setServerUrl('');
    expect(getServerUrl()).toBe('');
  });

  it('directs API calls to the override host', async () => {
    setServerUrl('http://10.0.0.4:3010');
    const seen: string[] = [];
    global.fetch = jest.fn().mockImplementation((url: string) => {
      seen.push(url);
      return Promise.resolve({ ok: true, json: async () => ({ ok: true, sessions: [] }) });
    }) as never;
    await listSessions();
    expect(seen[0]).toBe('http://10.0.0.4:3010/api/sessions');
  });

  it('rewrites a loopback relayUrl to the override host, keeping the relay port', async () => {
    setServerUrl('http://10.0.0.4:3010');
    global.fetch = jest.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, relayUrl: 'http://localhost:3010', status: 'open' }),
    }) as never;
    const res = await openSession('abc');
    expect(res.relayUrl).toBe('http://10.0.0.4:3010');
  });
});

it('createSessionFromYaml parses the files client-side and posts a JSON baseline', async () => {
  const captured: { url?: string; body?: unknown } = {};
  global.fetch = jest.fn().mockImplementation((url: string, init: RequestInit) => {
    captured.url = url;
    captured.body = init.body;
    return Promise.resolve({ ok: true, json: async () => ({ ok: true, sessionId: 'new', ownerToken: 'tok' }) });
  }) as never;
  const res = await createSessionFromYaml(
    'S',
    new File(['plan_range: { start_date: "2026-01-01", end_date: "2026-01-31" }'], 'Schedule.yaml'),
    new File(['worker_list: []'], 'EnvConfig.yaml'),
  );
  expect(res).toEqual({ sessionId: 'new', ownerToken: 'tok' });
  expect(captured.url).toMatch(/\/api\/sessions$/);
  const body = JSON.parse(captured.body as string);
  expect(body.name).toBe('S');
  expect(body.currentView).toBe('worker');
  expect(body.schedule.planRange).toEqual({ startDate: '2026-01-01', endDate: '2026-01-31' });
  expect(body.envConfig).toBeDefined();
});

it('fetchSessionName resolves the name for a real session', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, session: { name: 'Weekly Plan' } }) }) as never;
  expect(await fetchSessionName('abc123')).toBe('Weekly Plan');
});

it('fetchSessionName resolves null for an unknown session', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: false, json: async () => ({ ok: false }) }) as never;
  expect(await fetchSessionName('nope')).toBeNull();
});
