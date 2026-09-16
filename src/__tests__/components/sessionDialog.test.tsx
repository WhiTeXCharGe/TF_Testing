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

const FIXTURE_SCHEDULE = { planRange: { startDate: '2026-01-01', endDate: '2026-01-31' }, workflowTaskList: [], assignmentList: [] };
const FIXTURE_ENV_CONFIG = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

function Harness({ session, kind, loadedGantt }: { session?: SessionState; kind: SessionDialogKind; loadedGantt?: boolean }) {
  const { dispatch } = useAppContext();
  useEffect(() => {
    // LOAD_FILES first — the wrapped dispatch blocks it once a session is
    // active (stateRef reflects the SET_SESSION below only after a render),
    // so this order matters when a test combines both.
    if (loadedGantt) {
      dispatch({
        type: 'LOAD_FILES',
        payload: {
          schedule: FIXTURE_SCHEDULE, envConfig: FIXTURE_ENV_CONFIG, envPath: 'Env.yaml', schedulePath: 'Sched.yaml',
        },
      });
    }
    if (session) dispatch({ type: 'SET_SESSION', payload: session });
    dispatch({ type: 'OPEN_SESSION_DIALOG', payload: kind });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderDialog(props: { session?: SessionState; kind: SessionDialogKind; loadedGantt?: boolean }) {
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
  mockedCollab.getServerUrl.mockReturnValue('');
  mockedCollab.fetchLanAddresses.mockResolvedValue([]);
  mockedCollab.fetchLanHosts.mockResolvedValue([]);
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

    await userEvent.type(screen.getByPlaceholderText('ニックネームを入力'), 'Carol');
    expect(joinBtn).toBeDisabled(); // still no selection

    await userEvent.click(screen.getByText('Weekly Plan'));
    expect(joinBtn).toBeEnabled();
  });

  it('joins the selected session and remembers the display name', async () => {
    renderDialog({ kind: 'join' });
    await screen.findByText('Weekly Plan');
    await userEvent.type(screen.getByPlaceholderText('ニックネームを入力'), 'Carol');
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

  it('has no visible or enterable server-address field anywhere in the dialog', async () => {
    mockedCollab.getServerUrl.mockReturnValue('http://192.168.1.9:3010');
    renderDialog({ kind: 'join' });
    await screen.findByText('Weekly Plan');
    expect(screen.queryByDisplayValue('http://192.168.1.9:3010')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/空欄 = このPC/)).not.toBeInTheDocument();
    expect(screen.queryByText(/接続先サーバー/)).not.toBeInTheDocument();
  });

  it('shows discovered LAN hosts as plain-name buttons when this PC has nothing and several others are found; picking one switches silently', async () => {
    mockedCollab.listSessions.mockResolvedValue([]);
    mockedCollab.fetchLanHosts.mockResolvedValue([
      { name: 'DESKTOP-HOST', url: 'http://192.168.1.9:3010', lastSeenAt: 1 },
      { name: 'OTHER-HOST', url: 'http://192.168.1.10:3010', lastSeenAt: 1 },
    ]);
    renderDialog({ kind: 'join' });
    const chip = await screen.findByRole('button', { name: 'DESKTOP-HOST' });
    await userEvent.click(chip);
    expect(mockedCollab.setServerUrl).toHaveBeenCalledWith('http://192.168.1.9:3010');
  });

  it('auto-selects the only discovered host when this PC has no sessions of its own', async () => {
    mockedCollab.listSessions.mockResolvedValue([]);
    mockedCollab.fetchLanHosts.mockResolvedValue([
      { name: 'ONLY-ONE', url: 'http://192.168.1.9:3010', lastSeenAt: 1 },
    ]);
    renderDialog({ kind: 'join' });
    await waitFor(() => expect(mockedCollab.setServerUrl).toHaveBeenCalledWith('http://192.168.1.9:3010'));
  });

  it('does NOT auto-select in the create dialog (a host should stay on 空欄 = このPC)', async () => {
    mockedCollab.fetchLanHosts.mockResolvedValue([
      { name: 'ONLY-ONE', url: 'http://192.168.1.9:3010', lastSeenAt: 1 },
    ]);
    renderDialog({ kind: 'create' });
    await screen.findByText('EnvConfig YAML');
    await new Promise((r) => setTimeout(r, 50));
    expect(mockedCollab.setServerUrl).not.toHaveBeenCalled();
  });
});

describe('create dialog', () => {
  it('shows the EnvConfig file field before the Schedule file field', () => {
    renderDialog({ kind: 'create' });
    const box = screen.getByText('オンラインセッションを作成').parentElement as HTMLElement;
    const labels = within(box).getAllByText(/EnvConfig YAML|スケジュール YAML/).map((el) => el.textContent);
    expect(labels).toEqual(['EnvConfig YAML', 'スケジュール YAML']);
  });

  it('with no Gantt open: 現在のガントで開く is disabled, and 新しいガントをインポート is selected with file fields visible', () => {
    renderDialog({ kind: 'create' });
    expect(screen.getByLabelText('現在のガントで開く')).toBeDisabled();
    expect(screen.getByLabelText('新しいガントをインポート')).toBeChecked();
    expect(screen.getByText('EnvConfig YAML')).toBeInTheDocument();
  });

  it('with a Gantt already open: 現在のガントで開く is selected by default and the file fields are hidden', () => {
    renderDialog({ kind: 'create', loadedGantt: true });
    expect(screen.getByLabelText('現在のガントで開く')).toBeEnabled();
    expect(screen.getByLabelText('現在のガントで開く')).toBeChecked();
    expect(screen.queryByText('EnvConfig YAML')).not.toBeInTheDocument();
  });

  it('switching to 新しいガントをインポート reveals the file fields', async () => {
    renderDialog({ kind: 'create', loadedGantt: true });
    await userEvent.click(screen.getByLabelText('新しいガントをインポート'));
    expect(screen.getByText('EnvConfig YAML')).toBeInTheDocument();
  });

  it('creates from the current Gantt (no files) when that tab is submitted', async () => {
    mockedCollab.createSessionFromState.mockResolvedValue({ sessionId: 's9', ownerToken: 'tok' });
    renderDialog({ kind: 'create', loadedGantt: true });
    await userEvent.type(screen.getByPlaceholderText('ニックネームを入力'), 'Carol');
    await userEvent.type(screen.getByPlaceholderText('セッション名を入力'), 'Weekly Plan');
    await userEvent.click(screen.getByRole('button', { name: '作成して開始' }));
    await waitFor(() => expect(mockedCollab.createSessionFromState).toHaveBeenCalledWith(
      'Weekly Plan', expect.objectContaining({ schedule: FIXTURE_SCHEDULE, envConfig: FIXTURE_ENV_CONFIG }),
    ));
    expect(mockedCollab.createSessionFromYaml).not.toHaveBeenCalled();
  });
});

describe('info dialog', () => {
  // Participants are shown on hover over "n人が参加中" in the menu bar
  // (see menuBar.test.tsx), and lock/unlock is now a direct 共同編集 menu
  // action (also menuBar.test.tsx) — this dialog is just a read-only
  // name + status readout now.
  it('shows the session name and status, with no participant list or lock/unlock buttons', async () => {
    renderDialog({ session: activeSession(), kind: 'info' });
    expect(await screen.findByText(/My Session/)).toBeInTheDocument();
    expect(screen.getByText('開催中')).toBeInTheDocument();
    expect(screen.queryByText('Alice')).not.toBeInTheDocument();
    expect(screen.queryByText('ロックする')).not.toBeInTheDocument();
    expect(screen.queryByText('ロック解除')).not.toBeInTheDocument();
  });

  it('shows the locked explanation banner when the session is locked', async () => {
    renderDialog({ session: activeSession({ status: 'lock' }), kind: 'info' });
    expect(await screen.findByText('このセッションはロックされています（閲覧のみ）')).toBeInTheDocument();
  });
});

describe('update dialog (locked-session data replace)', () => {
  it('defaults to 現在のガントで開く and submitting sends the current schedule/envConfig', async () => {
    renderDialog({ session: activeSession({ status: 'lock' }), kind: 'update', loadedGantt: true });
    expect(await screen.findByLabelText('現在のガントで開く')).toBeChecked();

    await userEvent.click(screen.getByRole('button', { name: '更新する' }));

    await waitFor(() => expect(mockedCollab.sendCollabSessionUpdate).toHaveBeenCalledWith({
      schedule: FIXTURE_SCHEDULE, envConfig: FIXTURE_ENV_CONFIG, currentView: 'worker',
    }));
  });

  it('switching to 新しいガントをインポート shows the file fields, and submitting sends the parsed baseline', async () => {
    const parsed = { schedule: FIXTURE_SCHEDULE, envConfig: FIXTURE_ENV_CONFIG, currentView: 'worker' as const };
    mockedCollab.parseYamlBaseline.mockResolvedValue(parsed);
    renderDialog({ session: activeSession({ status: 'lock' }), kind: 'update', loadedGantt: true });

    await userEvent.click(await screen.findByLabelText('新しいガントをインポート'));
    expect(screen.getByText('EnvConfig YAML')).toBeInTheDocument();

    const fileInputs = document.querySelectorAll('input[type=file]');
    await userEvent.upload(fileInputs[0] as HTMLInputElement, new File(['b: 2'], 'EnvConfig.yaml'));
    await userEvent.upload(fileInputs[1] as HTMLInputElement, new File(['a: 1'], 'Schedule.yaml'));

    await userEvent.click(screen.getByRole('button', { name: '更新する' }));

    await waitFor(() => expect(mockedCollab.sendCollabSessionUpdate).toHaveBeenCalledWith(parsed));
  });
});
