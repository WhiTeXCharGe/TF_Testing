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

// GanttPage is stubbed so this file tests exactly one thing: that AppContent
// renders the shared shell for every role. Read-only button-level behavior
// (Toolbar, UndoRedoButtons, useKeyboardShortcuts) is proven end-to-end
// against the real page by cypress/e2e/06_viewer_parity.cy.ts instead.
jest.mock('../pages/GanttPage', () => ({ GanttPage: () => <div data-testid="editor-page" /> }));
jest.mock('../hooks/useKeyboardShortcuts', () => ({ useKeyboardShortcuts: jest.fn() }));
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
    onSyncInit('Test Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });
});

it('renders the shared editor shell in solo mode (no session)', () => {
  renderApp();
  expect(screen.getByTestId('editor-page')).toBeInTheDocument();
});

it('renders the shared editor shell for an edit-role session', async () => {
  renderApp();
  await userEvent.click(screen.getByText('join-edit'));
  await waitFor(() => expect(screen.getByTestId('editor-page')).toBeInTheDocument());
});

// The regression this file exists for: joining as 閲覧のみ from inside the app
// (the SessionDialog path) used to leave the fully interactive editor on
// screen, because read-only rendering was decided from the URL's ?role=view
// rather than from the session's own role. Now there is only ever one shell,
// so the assertion is simply that it renders here too — read-only-ness is a
// property of gating inside that shell, not of which shell got picked.
it('renders the same shared editor shell for a view-role session joined from inside the app', async () => {
  renderApp();
  await userEvent.click(screen.getByText('join-view'));
  await waitFor(() => expect(screen.getByTestId('editor-page')).toBeInTheDocument());
});
