/**
 * @jest-environment jsdom
 */
import { useEffect } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { SessionDialog } from '../../components/Dialogs/SessionDialog';
import { SessionState } from '../../types/appState';
import * as collabService from '../../services/collabService';

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

function activeSession(over: Partial<SessionState> = {}): SessionState {
  return {
    id: 's1', name: 'My Session', role: 'edit', connectionStatus: 'connected',
    status: 'open', participants: [
      { id: 'p1', name: 'Alice', role: 'edit' },
      { id: 'p2', name: 'Bob', role: 'view' },
    ],
    ...over,
  };
}

function Harness({ session, tab }: { session?: SessionState; tab?: 'list' | 'create' }) {
  const { dispatch } = useAppContext();
  useEffect(() => {
    if (session) dispatch({ type: 'SET_SESSION', payload: session });
    dispatch({ type: 'OPEN_SESSION_DIALOG', payload: tab ?? 'list' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderDialog(props: { session?: SessionState; tab?: 'list' | 'create' } = {}) {
  return render(
    <AppProvider>
      <Harness {...props} />
      <SessionDialog />
    </AppProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
  mockedCollab.listSessions.mockResolvedValue([
    { id: 's1', name: 'Weekly Plan', status: 'open', createdAt: 1, lastActivityAt: 2, participantCount: 3 },
    { id: 's2', name: 'Locked One', status: 'lock', createdAt: 1, lastActivityAt: 2, participantCount: 1 },
  ]);
  mockedCollab.openSession.mockResolvedValue({ relayUrl: 'http://relay:4010', status: 'open' });
  mockedCollab.joinCollabRoom.mockImplementation(() => () => {});
});

it('lists online sessions with their status', async () => {
  renderDialog();
  expect(await screen.findByText('Weekly Plan')).toBeInTheDocument();
  expect(screen.getByText('Locked One')).toBeInTheDocument();
  expect(screen.getByText('開催中')).toBeInTheDocument();
  expect(screen.getByText('ロック中')).toBeInTheDocument();
});

it('opens a session from the list, passing the display name and role', async () => {
  renderDialog();
  await screen.findByText('Weekly Plan');
  await userEvent.type(screen.getByPlaceholderText('表示名を入力'), 'Carol');
  await userEvent.click(screen.getAllByText('開く')[0]);
  await waitFor(() => expect(mockedCollab.openSession).toHaveBeenCalledWith('s1'));
});

it('shows every participant with their localized role in the active-session panel', async () => {
  renderDialog({ session: activeSession() });
  expect(await screen.findByText('Alice')).toBeInTheDocument();
  expect(screen.getByText('Bob')).toBeInTheDocument();
  expect(screen.getByText('編集者')).toBeInTheDocument();
  expect(screen.getByText('閲覧者')).toBeInTheDocument();
});

it('shows the ロックする button only when the participant holds an owner token', async () => {
  const { unmount } = renderDialog({ session: activeSession({ ownerToken: 'tok' }) });
  expect(await screen.findByText('ロックする')).toBeInTheDocument();
  unmount();

  renderDialog({ session: activeSession({ ownerToken: undefined }) });
  await screen.findByText('Alice');
  expect(screen.queryByText('ロックする')).not.toBeInTheDocument();
});

it('an owner unlocking a locked session calls unlockSession', async () => {
  renderDialog({ session: activeSession({ ownerToken: 'tok', status: 'lock' }) });
  const btn = await screen.findByText('ロック解除');
  await userEvent.click(btn);
  expect(mockedCollab.sendCollabUnlock).toHaveBeenCalled();
});
