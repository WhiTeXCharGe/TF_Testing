/**
 * @jest-environment jsdom
 */
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import * as collabService from '../../services/collabService';
import type { JoinCallbacks } from '../../services/collabService';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { UI } from '../../config/uiText';

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [],
  assignmentList: [],
};
const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

let capturedApi: ReturnType<typeof useAppContext> | null = null;

// The new joinCollabRoom takes (sessionId, name, role, isCreator, relayUrl,
// ownerToken, cb). This helper mocks it with a supplied driver that gets the
// isCreator flag and the callbacks bag.
function mockJoin(driver: (isCreator: boolean, cb: JoinCallbacks) => void) {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, isCreator, _relay, _token, cb) => {
    driver(isCreator, cb);
    return () => {};
  });
}

function TestConsumer() {
  const ctx = useAppContext();
  capturedApi = ctx;
  const { state, dispatch, startCollabSession, joinCollabSession, leaveCollabSession } = ctx;
  return (
    <div>
      <div data-testid="schedule-start">{state.schedule?.planRange.startDate ?? 'none'}</div>
      <div data-testid="session-role">{state.session?.role ?? 'none'}</div>
      <div data-testid="session-name">{state.session?.name ?? 'none'}</div>
      <div data-testid="session-status">{state.session?.status ?? 'none'}</div>
      <div data-testid="error-message">{state.errorMessage ?? 'none'}</div>
      <button onClick={() => dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } })}>load</button>
      <button onClick={() => dispatch({ type: 'LOAD_FILES', payload: { schedule: { ...SCHEDULE, planRange: { startDate: '2030-01-01', endDate: '2030-01-31' } }, envConfig: ENV_CONFIG, envPath: 'e2.yaml', schedulePath: 's2.yaml' } })}>load-other</button>
      <button onClick={() => dispatch({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } })}>edit</button>
      <button onClick={() => dispatch({ type: 'UNDO' })}>undo</button>
      <button onClick={() => dispatch({ type: 'REDO' })}>redo</button>
      <button onClick={() => dispatch({ type: 'TOGGLE_FLIGHT_STINTS' })}>toggle-local</button>
      <button onClick={() => void startCollabSession('Alice', 'My Session')}>start</button>
      <button onClick={() => void joinCollabSession('abc', 'Bob', 'edit')}>join</button>
      <button onClick={() => void joinCollabSession('abc', 'Carol', 'view')}>join-view</button>
      <button onClick={() => leaveCollabSession()}>leave</button>
    </div>
  );
}

function renderApp() {
  return render(<AppProvider><TestConsumer /></AppProvider>);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
  mockedCollab.createSessionFromState.mockResolvedValue({ sessionId: 's1', ownerToken: 'owner-tok' });
  mockedCollab.createSessionFromYaml.mockResolvedValue({ sessionId: 's1', ownerToken: 'owner-tok' });
  mockedCollab.openSession.mockResolvedValue({ relayUrl: 'http://relay:4010', status: 'open' });
});

it('forwards a syncable action to the server while in an edit session, but not a local-only one', async () => {
  mockJoin((_isCreator, cb) => cb.onStatusChange('connected'));

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await act(async () => { await userEvent.click(screen.getByText('start')); });

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1);
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('UPDATE_PLAN_RANGE', { startDate: '2026-02-01', endDate: '2026-02-28' });

  await userEvent.click(screen.getByText('toggle-local'));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1);
});

it('applies a remote action via the raw dispatch without forwarding it back to the server', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockJoin((_isCreator, cb) => {
    cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = cb.onAction;
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } }));

  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-03-01'));
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
});

it('skips re-applying the baseline for the session creator', async () => {
  mockJoin((isCreator, cb) => {
    if (!isCreator) {
      cb.onSyncInit('Mock Session', { schedule: { ...SCHEDULE, planRange: { startDate: '1999-01-01', endDate: '1999-01-02' } }, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    }
  });

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await act(async () => { await userEvent.click(screen.getByText('start')); });

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('session-role')).toHaveTextContent('edit');
});

it('does not forward actions when joined as a view-only participant', async () => {
  mockJoin((_isCreator, cb) => cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []));

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join-view')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));
  expect(screen.getByTestId('session-role')).toHaveTextContent('view');

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));

  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
});

it('ignores an inbound action whose type is not syncable', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockJoin((_isCreator, cb) => {
    cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = cb.onAction;
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({ type: 'SET_ERROR', payload: 'injected by a peer' }));

  expect(screen.getByTestId('error-message')).toHaveTextContent('none');
});

it('ignores an inbound LOAD_FILES, which would otherwise bypass the mid-session load block', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockJoin((_isCreator, cb) => {
    cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = cb.onAction;
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({
    type: 'LOAD_FILES',
    payload: { schedule: { ...SCHEDULE, planRange: { startDate: '2030-01-01', endDate: '2030-01-31' } }, envConfig: ENV_CONFIG, envPath: 'x.yaml', schedulePath: 'y.yaml' },
  }));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
});

it('filters non-syncable action types out of the sync-init log replay too', async () => {
  mockJoin((_isCreator, cb) => cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, [
    { seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-04-01', endDate: '2026-04-30' } },
    { seq: 2, type: 'SET_ERROR', payload: 'injected into the log' },
  ]));

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });

  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-04-01'));
  expect(screen.getByTestId('error-message')).toHaveTextContent('none');
});

it('rejects starting a session when no schedule is loaded yet', async () => {
  renderApp();
  await expect(capturedApi!.startCollabSession('Alice', 'My Session')).rejects.toThrow(UI.collabNoScheduleError);
});

it('lets a participant leave a session, clearing session state', async () => {
  mockJoin((_isCreator, cb) => cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []));

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('edit'));

  await userEvent.click(screen.getByText('leave'));

  expect(screen.getByTestId('session-role')).toHaveTextContent('none');
});

describe('leave-time checkpoint (last participant hands the server a final snapshot)', () => {
  it('sends one when an editor leaves as the last (only) participant', async () => {
    mockJoin((_isCreator, cb) => {
      cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      cb.onPresence([{ id: 'me', name: 'Bob', role: 'edit' }]);
    });
    renderApp();
    await act(async () => { await userEvent.click(screen.getByText('join')); });
    await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('edit'));

    await userEvent.click(screen.getByText('leave'));

    expect(mockedCollab.sendCollabCheckpoint).toHaveBeenCalledWith({
      schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker',
    });
  });

  it('does not send one when other participants remain', async () => {
    mockJoin((_isCreator, cb) => {
      cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      cb.onPresence([{ id: 'me', name: 'Bob', role: 'edit' }, { id: 'other', name: 'Alice', role: 'edit' }]);
    });
    renderApp();
    await act(async () => { await userEvent.click(screen.getByText('join')); });
    await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('edit'));

    await userEvent.click(screen.getByText('leave'));

    expect(mockedCollab.sendCollabCheckpoint).not.toHaveBeenCalled();
  });

  it('does not send one for a view-only participant leaving last', async () => {
    mockJoin((_isCreator, cb) => {
      cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      cb.onPresence([{ id: 'me', name: 'Carol', role: 'view' }]);
    });
    renderApp();
    await act(async () => { await userEvent.click(screen.getByText('join-view')); });
    await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('view'));

    await userEvent.click(screen.getByText('leave'));

    expect(mockedCollab.sendCollabCheckpoint).not.toHaveBeenCalled();
  });
});

it('blocks LOAD_FILES while a session is active, regardless of role, and surfaces an error', async () => {
  mockJoin((_isCreator, cb) => cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []));

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join-view')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByText('load-other'));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('error-message')).toHaveTextContent(UI.collabActiveLoadBlockedError);
});

it('still allows LOAD_FILES normally when no session is active (solo mode is unaffected)', async () => {
  renderApp();
  await userEvent.click(screen.getByText('load'));
  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('error-message')).toHaveTextContent('none');
});

it('sets the session name from the sync-init reply when joining', async () => {
  mockJoin((_isCreator, cb) => cb.onSyncInit('Joined Session Name', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []));

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('session-name')).toHaveTextContent('Joined Session Name'));
});

it('flips to read-only and blocks edits when a session-status lock arrives', async () => {
  let cbRef: JoinCallbacks | null = null;
  mockJoin((_isCreator, cb) => { cbRef = cb; cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []); cb.onStatusChange('connected'); });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('edit'));

  act(() => cbRef!.onSessionStatus('lock'));
  await waitFor(() => expect(screen.getByTestId('session-status')).toHaveTextContent('lock'));

  mockedCollab.sendCollabAction.mockClear();
  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
  expect(screen.getByTestId('error-message')).toHaveTextContent(UI.collabLockedEditBlockedError);
});

describe('own-action, conflict-aware undo/redo', () => {
  const RICH_SCHEDULE: ScheduleData = {
    planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
    workflowTaskList: [
      {
        id: 'wt1', workflow: 'wf1',
        phaseTaskList: [
          {
            id: 'pt1', phase: 'p1', startDate: '2026-01-01', endDate: '2026-01-15',
            operationTaskList: [{ id: 'ot1', operation: 'op1', workloadHours: 10, colorCode: 'red' }],
          },
        ],
      },
    ],
    assignmentList: [
      { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-01', endDate: '2026-01-05', planFlexibility: 'Flexible', workDateList: [] },
    ],
  };

  it('undoes my own later edits freely, but blocks on an edit whose object someone else touched since — reproduces the design doc scenario', async () => {
    let capturedOnAction: ((a: { type: string; payload: unknown }) => void) | null = null;
    mockJoin((_isCreator, cb) => {
      cb.onSyncInit('Mock Session', { schedule: RICH_SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      capturedOnAction = cb.onAction;
      cb.onStatusChange('connected');
    });
    renderApp();
    await act(async () => { await userEvent.click(screen.getByText('join')); });
    await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

    // "userA": P1 move O1, P2 move O1 again, P3 color O4 (a different object), P4 add O5 (a brand-new assignment)
    act(() => capturedApi!.dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-02' } } })); // P1
    act(() => capturedApi!.dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-03' } } })); // P2
    act(() => capturedApi!.dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: 'ot1', colorCode: 'green' } })); // P3
    act(() => capturedApi!.dispatch({ type: 'ADD_ASSIGNMENT', payload: { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-10', endDate: '2026-01-11', planFlexibility: 'Flexible', workDateList: [] } })); // P4

    // "userB": a remote edit to the SAME object as P2 (O1), arriving over the wire via the real applyRemoteAction path.
    act(() => capturedOnAction!({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-20' } } }));

    // Undo P4 (add O5) — untouched, must succeed: assignment count back to 1.
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList).toHaveLength(1);

    // Undo P3 (O4 color) — untouched, must succeed.
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.workflowTaskList[0].phaseTaskList[0].operationTaskList[0].colorCode).toBe('red');

    // Undo P2 (O1) — but "userB" touched O1 since — must be BLOCKED, nothing changes.
    const startDateNow = capturedApi!.state.schedule!.assignmentList[0].startDate;
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList[0].startDate).toBe(startDateNow);
    expect(screen.getByTestId('error-message')).toHaveTextContent(UI.undoBlockedError);

    // Clicking Undo again re-checks the SAME entry — blocked again, no silent cascade to P1.
    act(() => capturedApi!.dispatch({ type: 'SET_ERROR', payload: null }));
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList[0].startDate).toBe(startDateNow);
    expect(screen.getByTestId('error-message')).toHaveTextContent(UI.undoBlockedError);
  });

  it('a safe undo forwards the revert action to the server, same as any normal edit', async () => {
    mockJoin((_isCreator, cb) => {
      cb.onSyncInit('Mock Session', { schedule: RICH_SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      cb.onStatusChange('connected');
    });
    renderApp();
    await act(async () => { await userEvent.click(screen.getByText('join')); });
    await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

    act(() => capturedApi!.dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: 'ot1', colorCode: 'blue' } }));
    mockedCollab.sendCollabAction.mockClear();
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('UPDATE_OPERATION_TASK_COLOR', { operationTaskId: 'ot1', colorCode: 'red' });
  });

  it('solo mode (no session) keeps working: undo/redo apply and revert locally with no server involvement', async () => {
    renderApp();
    await userEvent.click(screen.getByText('load')); // LOAD_FILES with the shared empty SCHEDULE/ENV_CONFIG fixtures — solo mode doesn't need RICH_SCHEDULE
    await userEvent.click(screen.getByRole('button', { name: 'edit' })); // UPDATE_PLAN_RANGE, from the existing button
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-02-01');
    await userEvent.click(screen.getByText('undo'));
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
    await userEvent.click(screen.getByText('redo'));
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-02-01');
    expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
  });
});
