/**
 * @jest-environment jsdom
 */
import { useEffect } from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { SessionDialog } from '../../components/Dialogs/SessionDialog';
import { SessionDialogKind, SessionState, SessionSummary } from '../../types/appState';
import * as collabService from '../../services/collabService';

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

const summary = (over: Partial<SessionSummary> = {}): SessionSummary => ({
  id: 's1', name: 'Weekly Plan', status: 'open', createdAt: 1000, lastActivityAt: 2000,
  lastJoinAt: 5000, participantCount: 3, ...over,
});

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

function Harness({ session, kind }: { session?: SessionState; kind: SessionDialogKind }) {
  const { dispatch } = useAppContext();
  useEffect(() => {
    if (session) dispatch({ type: 'SET_SESSION', payload: session });
    dispatch({ type: 'OPEN_SESSION_DIALOG', payload: kind });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderDialog(props: { session?: SessionState; kind: SessionDialogKind }) {
  return render(
    <AppProvider>
      <Harness {...props} />
      <SessionDialog />
    </AppProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
  mockedCollab.listSessions.mockResolvedValue([
    summary({ id: 's1', name: 'Weekly Plan', status: 'open', lastJoinAt: 5000 }),
    summary({ id: 's2', name: 'Locked One', status: 'lock', lastJoinAt: 9000, participantCount: 1 }),
  ]);
  mockedCollab.openSession.mockResolvedValue({ relayUrl: 'http://relay:4010', status: 'open' });
  mockedCollab.joinCollabRoom.mockImplementation(() => () => {});
});

describe('join dialog', () => {
  it('lists sessions with status, most-recently-joined first', async () => {
    renderDialog({ kind: 'join' });
    await screen.findByText('Weekly Plan');
    const rows = screen.getAllByText(/Plan|Locked One/).map((el) => el.textContent);
    expect(rows[0]).toBe('Locked One'); // lastJoinAt 9000 > 5000
    expect(screen.getByText('開催中')).toBeInTheDocument();
    expect(screen.getByText('ロック中')).toBeInTheDocument();
  });

  it('the 参加 button is disabled until a row is selected and a name is entered', async () => {
    renderDialog({ kind: 'join' });
    await screen.findByText('Weekly Plan');
    const joinBtn = screen.getByRole('button', { name: '参加' });
    expect(joinBtn).toBeDisabled();

    await userEvent.type(screen.getByPlaceholderText('表示名を入力'), 'Carol');
    expect(joinBtn).toBeDisabled(); // still no selection

    await userEvent.click(screen.getByText('Weekly Plan'));
    expect(joinBtn).toBeEnabled();
  });

  it('joins the selected session and remembers the display name', async () => {
    renderDialog({ kind: 'join' });
    await screen.findByText('Weekly Plan');
    await userEvent.type(screen.getByPlaceholderText('表示名を入力'), 'Carol');
    await userEvent.click(screen.getByText('Weekly Plan'));
    await userEvent.click(screen.getByRole('button', { name: '参加' }));
    await waitFor(() => expect(mockedCollab.openSession).toHaveBeenCalledWith('s1'));
    expect(localStorage.getItem('gantt.collab.displayName')).toBe('Carol');
  });

  it('pre-fills the display name from a previous session', async () => {
    localStorage.setItem('gantt.collab.displayName', 'Dave');
    renderDialog({ kind: 'join' });
    expect(await screen.findByDisplayValue('Dave')).toBeInTheDocument();
  });
});

describe('create dialog', () => {
  it('shows the EnvConfig file field before the Schedule file field', () => {
    renderDialog({ kind: 'create' });
    const box = screen.getByText('オンラインセッションを作成').parentElement as HTMLElement;
    const labels = within(box).getAllByText(/EnvConfig YAML|スケジュール YAML/).map((el) => el.textContent);
    expect(labels).toEqual(['EnvConfig YAML', 'スケジュール YAML']);
  });
});

describe('info dialog', () => {
  it('lists participants with their localized role', async () => {
    renderDialog({ session: activeSession(), kind: 'info' });
    expect(await screen.findByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('編集者')).toBeInTheDocument();
    expect(screen.getByText('閲覧者')).toBeInTheDocument();
  });

  it('shows the lock button to any participant (no owner token needed)', async () => {
    renderDialog({ session: activeSession({ ownerToken: undefined }), kind: 'info' });
    expect(await screen.findByText('ロックする')).toBeInTheDocument();
  });

  it('unlocking a locked session calls unlockSession', async () => {
    renderDialog({ session: activeSession({ status: 'lock' }), kind: 'info' });
    await userEvent.click(await screen.findByText('ロック解除'));
    expect(mockedCollab.sendCollabUnlock).toHaveBeenCalled();
  });
});
