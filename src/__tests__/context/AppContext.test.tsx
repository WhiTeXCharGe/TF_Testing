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

it('forwards undo as the resulting SET_SCHEDULE snapshot, not the bare UNDO token', async () => {
  mockJoin((_isCreator, cb) => {
    cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    cb.onStatusChange('connected');
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  mockedCollab.sendCollabAction.mockClear();

  await userEvent.click(screen.getByText('undo'));

  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('SET_SCHEDULE', SCHEDULE);
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalledWith('UNDO', undefined);
});

it('forwards redo as the resulting SET_SCHEDULE snapshot, not the bare REDO token', async () => {
  mockJoin((_isCreator, cb) => {
    cb.onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    cb.onStatusChange('connected');
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  await userEvent.click(screen.getByText('undo'));
  mockedCollab.sendCollabAction.mockClear();

  await userEvent.click(screen.getByText('redo'));

  const editedSchedule: ScheduleData = { ...SCHEDULE, planRange: { startDate: '2026-02-01', endDate: '2026-02-28' } };
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('SET_SCHEDULE', editedSchedule);
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalledWith('REDO', undefined);
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
