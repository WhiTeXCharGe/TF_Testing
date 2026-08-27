/**
 * @jest-environment jsdom
 */
import { io } from 'socket.io-client';
import { joinCollabRoom } from '../../services/collabService';
import { SessionBaseline } from '../../types/appState';

jest.mock('socket.io-client', () => ({ io: jest.fn() }));

type Handler = (payload: never) => void;

const BASELINE: SessionBaseline = {
  schedule: { planRange: { startDate: '2026-01-01', endDate: '2026-01-31' }, workflowTaskList: [], assignmentList: [] },
  envConfig: { workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [] },
  currentView: 'worker',
};

// Minimal stand-in for the socket.io client: just enough to capture the
// handlers joinCollabRoom registers so a test can fire 'sync-init' by hand.
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
// collabService keeps ONE module-level socket, cleared only by the disconnect
// its join returns. Tearing every join down here (rather than at the end of
// each test) keeps a failing assertion from leaving that singleton pointing at
// a stale fake and cascading into the next test.
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
  const onSyncInit = jest.fn();
  const onStatusChange = jest.fn();
  const disconnect = joinCollabRoom(
    's1', 'Alice', 'edit', isCreator,
    onSyncInit,
    jest.fn(), jest.fn(), onStatusChange,
  );
  openJoins.push(disconnect);
  return { onSyncInit, onStatusChange };
}

const SYNC_INIT_OK = { ok: true, baseline: BASELINE, actions: [], participants: [] };

it('skips the baseline replay for the creator on their first sync-init', () => {
  const { onSyncInit } = join(true);

  fakeSocket.fire('sync-init', SYNC_INIT_OK);

  expect(onSyncInit).not.toHaveBeenCalled(); // their local state already IS the baseline
});

// socket.io-client reconnects on its own and every reconnect re-emits 'join',
// producing another sync-init. If the creator kept skipping those they would
// permanently miss everything that happened while they were disconnected,
// while still reporting 'connected'.
it('replays the baseline for the creator on a later (reconnect) sync-init', () => {
  const { onSyncInit } = join(true);

  fakeSocket.fire('sync-init', SYNC_INIT_OK);      // initial
  fakeSocket.fire('sync-init', {                   // after a reconnect
    ...SYNC_INIT_OK,
    actions: [{ seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-05-01', endDate: '2026-05-31' } }],
  });

  expect(onSyncInit).toHaveBeenCalledTimes(1);
  expect(onSyncInit).toHaveBeenCalledWith(BASELINE, [
    { seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-05-01', endDate: '2026-05-31' } },
  ]);
});

it('replays the baseline for a non-creator on every sync-init', () => {
  const { onSyncInit } = join(false);

  fakeSocket.fire('sync-init', SYNC_INIT_OK);
  fakeSocket.fire('sync-init', SYNC_INIT_OK);

  expect(onSyncInit).toHaveBeenCalledTimes(2);
});

it('reports disconnected and replays nothing when sync-init comes back not-ok', () => {
  const { onSyncInit, onStatusChange } = join(false);

  fakeSocket.fire('sync-init', { ok: false });

  expect(onSyncInit).not.toHaveBeenCalled();
  expect(onStatusChange).toHaveBeenLastCalledWith('disconnected');
});
