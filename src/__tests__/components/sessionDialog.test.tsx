/**
 * @jest-environment jsdom
 *
 * Covers real-usage feedback: セッション情報 (the active-session info dialog)
 * should show the participant list, not just the count.
 */
import { useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { SessionDialog } from '../../components/Dialogs/SessionDialog';
import * as collabService from '../../services/collabService';

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

function Harness() {
  const { dispatch } = useAppContext();
  useEffect(() => {
    dispatch({
      type: 'SET_SESSION',
      payload: {
        id: 's1',
        name: 'My Session',
        role: 'edit',
        connectionStatus: 'connected',
        participants: [
          { id: 'p1', name: 'Alice', role: 'edit' },
          { id: 'p2', name: 'Bob', role: 'view' },
        ],
      },
    });
    dispatch({ type: 'OPEN_SESSION_DIALOG', payload: 'start' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.fetchCollabLink.mockResolvedValue('http://example.com/?session=s1');
});

it('shows every participant with their localized role in the session-info dialog', async () => {
  render(
    <AppProvider>
      <Harness />
      <SessionDialog />
    </AppProvider>,
  );

  expect(await screen.findByText('Alice')).toBeInTheDocument();
  expect(screen.getByText('Bob')).toBeInTheDocument();
  expect(screen.getByText('編集者')).toBeInTheDocument();
  expect(screen.getByText('閲覧者')).toBeInTheDocument();
});
