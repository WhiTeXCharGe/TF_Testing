/**
 * @jest-environment jsdom
 */
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import * as collabService from '../../services/collabService';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';

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

function TestConsumer() {
  const { state, dispatch, startCollabSession, joinCollabSession, leaveCollabSession } = useAppContext();
  return (
    <div>
      <div data-testid="schedule-start">{state.schedule?.planRange.startDate ?? 'none'}</div>
      <div data-testid="session-role">{state.session?.role ?? 'none'}</div>
      <button onClick={() => dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } })}>load</button>
      <button onClick={() => dispatch({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } })}>edit</button>
      <button onClick={() => dispatch({ type: 'UNDO' })}>undo</button>
      <button onClick={() => dispatch({ type: 'TOGGLE_FLIGHT_STINTS' })}>toggle-local</button>
      <button onClick={() => void startCollabSession('Alice')}>start</button>
      <button onClick={() => void joinCollabSession('abc', 'Bob', 'edit')}>join</button>
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
  mockedCollab.joinCollabRoom.mockReturnValue(() => {});

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
    onSyncInit({ schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
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
      onSyncInit({ schedule: { ...SCHEDULE, planRange: { startDate: '1999-01-01', endDate: '1999-01-02' } }, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
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
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit({ schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
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
