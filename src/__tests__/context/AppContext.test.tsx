/**
 * @jest-environment jsdom
 */
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import * as collabService from '../../services/collabService';
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

// Captured on every render so tests can call context functions directly
// (e.g. to assert on a rejected Promise) when a button's fire-and-forget
// `void fn(...)` onClick handler would otherwise swallow the rejection.
let capturedApi: ReturnType<typeof useAppContext> | null = null;

function TestConsumer() {
  const ctx = useAppContext();
  capturedApi = ctx;
  const { state, dispatch, startCollabSession, joinCollabSession, leaveCollabSession } = ctx;
  return (
    <div>
      <div data-testid="schedule-start">{state.schedule?.planRange.startDate ?? 'none'}</div>
      <div data-testid="session-role">{state.session?.role ?? 'none'}</div>
      <div data-testid="session-name">{state.session?.name ?? 'none'}</div>
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
});

it('forwards a syncable action to the server while in an edit session, but not a local-only one', async () => {
  mockedCollab.createCollabSession.mockResolvedValue('s1');
  mockedCollab.fetchCollabLink.mockResolvedValue('http://host/?session=s1&role=edit');
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, _onSyncInit, _onAction, _onPresence, onStatusChange) => {
    onStatusChange('connected');
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await userEvent.click(screen.getByText('start'));

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1);
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('UPDATE_PLAN_RANGE', { startDate: '2026-02-01', endDate: '2026-02-28' });

  await userEvent.click(screen.getByText('toggle-local'));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1); // still 1 — the local-only action wasn't forwarded
});

it('applies a remote action via the raw dispatch without forwarding it back to the server', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, onAction) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = onAction;
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } }));

  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-03-01'));
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
});

it('skips re-applying the baseline for the session creator', async () => {
  mockedCollab.createCollabSession.mockResolvedValue('s1');
  mockedCollab.fetchCollabLink.mockResolvedValue('http://host/?session=s1&role=edit');
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, isCreator, onSyncInit) => {
    if (!isCreator) {
      onSyncInit('Mock Session', { schedule: { ...SCHEDULE, planRange: { startDate: '1999-01-01', endDate: '1999-01-02' } }, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    }
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await userEvent.click(screen.getByText('start'));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('session-role')).toHaveTextContent('edit');
});

it('forwards undo as the resulting SET_SCHEDULE snapshot, not the bare UNDO token', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, _onAction, _onPresence, onStatusChange) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    onStatusChange('connected');
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));
  mockedCollab.sendCollabAction.mockClear();

  await userEvent.click(screen.getByText('undo'));

  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('SET_SCHEDULE', SCHEDULE);
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalledWith('UNDO', undefined);
});

it('forwards redo as the resulting SET_SCHEDULE snapshot, not the bare REDO token', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, _onAction, _onPresence, onStatusChange) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    onStatusChange('connected');
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByRole('button', { name: 'edit' })); // pushes SCHEDULE onto undoStack
  await userEvent.click(screen.getByText('undo')); // reverts to SCHEDULE, pushes the edited version onto redoStack
  mockedCollab.sendCollabAction.mockClear();

  await userEvent.click(screen.getByText('redo'));

  const editedSchedule: ScheduleData = { ...SCHEDULE, planRange: { startDate: '2026-02-01', endDate: '2026-02-28' } };
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('SET_SCHEDULE', editedSchedule);
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalledWith('REDO', undefined);
});

it('does not forward actions when joined as a view-only participant', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join-view'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));
  expect(screen.getByTestId('session-role')).toHaveTextContent('view');

  await userEvent.click(screen.getByRole('button', { name: 'edit' }));

  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
});

// The relay forwards whatever `type` string an edit-role peer emits without
// validating it, so inbound actions are untrusted: SYNCABLE_ACTION_TYPES has
// to gate the applying side too, not just the sending side.
it('ignores an inbound action whose type is not syncable', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, onAction) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = onAction;
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({ type: 'SET_ERROR', payload: 'injected by a peer' }));

  expect(screen.getByTestId('error-message')).toHaveTextContent('none');
});

// The mid-session load block lives in the OUTGOING wrapper only, so an
// inbound LOAD_FILES would otherwise sail straight past it and silently swap
// this participant's document — the exact corruption that block exists to stop.
it('ignores an inbound LOAD_FILES, which would otherwise bypass the mid-session load block', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, onAction) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = onAction;
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({
    type: 'LOAD_FILES',
    payload: { schedule: { ...SCHEDULE, planRange: { startDate: '2030-01-01', endDate: '2030-01-31' } }, envConfig: ENV_CONFIG, envPath: 'x.yaml', schedulePath: 'y.yaml' },
  }));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'); // unchanged
});

it('filters non-syncable action types out of the sync-init log replay too', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, [
      { seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-04-01', endDate: '2026-04-30' } },
      { seq: 2, type: 'SET_ERROR', payload: 'injected into the log' },
    ]);
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));

  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-04-01')); // real edit replayed
  expect(screen.getByTestId('error-message')).toHaveTextContent('none'); // the injected one was not
});

it('rejects starting a session when no schedule is loaded yet', async () => {
  renderApp();

  await expect(capturedApi!.startCollabSession('Alice', 'My Session')).rejects.toThrow(UI.collabNoScheduleError);
});

it('lets a participant leave a session, clearing session state', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join'));
  await waitFor(() => expect(screen.getByTestId('session-role')).toHaveTextContent('edit'));

  await userEvent.click(screen.getByText('leave'));

  expect(screen.getByTestId('session-role')).toHaveTextContent('none');
});

it('blocks LOAD_FILES while a session is active, regardless of role, and surfaces an error', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Mock Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('join-view'));
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByText('load-other'));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'); // unchanged — LOAD_FILES was blocked
  expect(screen.getByTestId('error-message')).toHaveTextContent(UI.collabActiveLoadBlockedError);
});

it('still allows LOAD_FILES normally when no session is active (solo mode is unaffected)', async () => {
  renderApp();

  await userEvent.click(screen.getByText('load'));

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('error-message')).toHaveTextContent('none');
});

it('sets the session name from the sync-init reply when joining', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Joined Session Name', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('session-name')).toHaveTextContent('Joined Session Name'));
});
