/**
 * @jest-environment jsdom
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContent } from '../App';
import { AppProvider, useAppContext } from '../context/AppContext';
import * as collabService from '../services/collabService';
import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';

// The two pages are stubbed so this file tests exactly one thing: which of
// them AppContent picks. Their real rendering is covered elsewhere.
jest.mock('../pages/GanttPage', () => ({ GanttPage: () => <div data-testid="editor-page" /> }));
jest.mock('../pages/ViewPage', () => ({ ViewPage: () => <div data-testid="view-page" /> }));

// Editor-only hooks. Asserting these never mount for a view participant is the
// point: useKeyboardShortcuts is what binds Ctrl+S (saving a document that has
// diverged from the shared one) and Ctrl+O.
const mockUseKeyboardShortcuts = jest.fn();
jest.mock('../hooks/useKeyboardShortcuts', () => ({ useKeyboardShortcuts: () => mockUseKeyboardShortcuts() }));
jest.mock('../hooks/useConstraintCheck', () => ({ useConstraintCheck: jest.fn() }));
jest.mock('../hooks/useIncomingGanttTransfer', () => ({ useIncomingGanttTransfer: jest.fn() }));

jest.mock('../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [],
  assignmentList: [],
};
const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

// Sits alongside AppContent inside the same provider so a test can join a
// session the same way the in-app SessionDialog does — no URL involved.
function JoinHarness() {
  const { joinCollabSession } = useAppContext();
  return (
    <>
      <button onClick={() => void joinCollabSession('abc', 'Carol', 'view')}>join-view</button>
      <button onClick={() => void joinCollabSession('abc', 'Bob', 'edit')}>join-edit</button>
    </>
  );
}

function renderApp() {
  return render(
    <AppProvider>
      <JoinHarness />
      <AppContent />
    </AppProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit({ schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });
});

it('renders the editor in solo mode (no session)', () => {
  renderApp();

  expect(screen.getByTestId('editor-page')).toBeInTheDocument();
  expect(screen.queryByTestId('view-page')).not.toBeInTheDocument();
});

it('renders the editor for an edit-role session', async () => {
  renderApp();

  await userEvent.click(screen.getByText('join-edit'));

  await waitFor(() => expect(screen.getByTestId('editor-page')).toBeInTheDocument());
  expect(screen.queryByTestId('view-page')).not.toBeInTheDocument();
});

// The regression this file exists for: joining as 閲覧のみ from inside the app
// (the SessionDialog path) used to leave the fully interactive editor on
// screen, because read-only rendering was decided from the URL's ?role=view
// rather than from the session's own role.
it('renders the read-only ViewPage for a view-role session joined from inside the app', async () => {
  renderApp();

  await userEvent.click(screen.getByText('join-view'));

  await waitFor(() => expect(screen.getByTestId('view-page')).toBeInTheDocument());
  expect(screen.queryByTestId('editor-page')).not.toBeInTheDocument();
});

it('does not mount the editor-only hooks (Ctrl+S / Ctrl+O) for a view-role session', async () => {
  renderApp();
  expect(mockUseKeyboardShortcuts).toHaveBeenCalled(); // solo: editor is up

  await userEvent.click(screen.getByText('join-view'));
  await waitFor(() => expect(screen.getByTestId('view-page')).toBeInTheDocument());

  mockUseKeyboardShortcuts.mockClear();
  // Re-render the tree; the view participant's shell must not call it again.
  await userEvent.click(screen.getByText('join-view'));
  expect(mockUseKeyboardShortcuts).not.toHaveBeenCalled();
});
