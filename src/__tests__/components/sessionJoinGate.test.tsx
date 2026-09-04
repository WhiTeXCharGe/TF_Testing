/**
 * @jest-environment jsdom
 *
 * Covers real-usage feedback: when a participant opens a session link
 * directly in a browser (App's ?session=<id> URL-param gate, distinct from
 * the in-app SessionDialog join flow), the prompt should name the session
 * ("「<name>」の参加") once it resolves, instead of a permanent generic
 * "このセッションに参加".
 */
import { render, screen, waitFor } from '@testing-library/react';
import App from '../../App';
import * as collabService from '../../services/collabService';

// App.tsx statically imports GanttPage (and, transitively, MenuBar's
// excelExportService -> exceljs, an ESM-only package Jest's CJS transform
// can't parse) even for this pre-join screen, which never renders it —
// mocked the same way App.test.tsx does for the shared editor shell.
jest.mock('../../pages/GanttPage', () => ({ GanttPage: () => <div data-testid="editor-page" /> }));
jest.mock('../../hooks/useKeyboardShortcuts', () => ({ useKeyboardShortcuts: jest.fn() }));
jest.mock('../../hooks/useConstraintCheck', () => ({ useConstraintCheck: jest.fn() }));
jest.mock('../../hooks/useIncomingGanttTransfer', () => ({ useIncomingGanttTransfer: jest.fn() }));

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

beforeEach(() => {
  jest.clearAllMocks();
  window.history.pushState({}, '', '/?session=abc123&role=edit');
});

afterEach(() => {
  window.history.pushState({}, '', '/');
});

it('shows the generic title while the session name is still resolving', () => {
  mockedCollab.fetchSessionName.mockReturnValue(new Promise(() => {})); // never resolves
  render(<App />);
  expect(screen.getByText('このセッションに参加')).toBeInTheDocument();
});

it('shows the resolved session name once the lookup succeeds', async () => {
  mockedCollab.fetchSessionName.mockResolvedValue('AAA');
  render(<App />);
  expect(await screen.findByText('「AAA」の参加')).toBeInTheDocument();
  expect(screen.queryByText('このセッションに参加')).not.toBeInTheDocument();
});

it('falls back to the generic title when the lookup fails to resolve a name', async () => {
  mockedCollab.fetchSessionName.mockResolvedValue(null);
  render(<App />);
  await waitFor(() => expect(mockedCollab.fetchSessionName).toHaveBeenCalledWith('abc123'));
  expect(screen.getByText('このセッションに参加')).toBeInTheDocument();
});
