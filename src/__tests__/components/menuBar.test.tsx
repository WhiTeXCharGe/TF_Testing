/**
 * @jest-environment jsdom
 *
 * Covers real-usage feedback: the participant list in the menu bar should
 * open on hover as well as click (it was click-only before).
 */
import { useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { MenuBar } from '../../components/Toolbar/MenuBar';
import * as collabService from '../../services/collabService';
import { SessionStatus } from '../../types/appState';

// excelExportService pulls in exceljs, an ESM-only package Jest's CJS
// transform can't parse even to auto-mock (auto-mocking still loads the
// real module to introspect its shape) — a factory mock avoids loading it
// at all, since export behavior isn't what this file tests.
jest.mock('../../services/excelExportService', () => ({ exportScheduleToExcel: jest.fn() }));
// lockSession/unlockSession go straight through collabService's socket
// wrapper — mock it so lock/unlock clicks are observable without a real
// connection.
jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

function Harness({ withSession = true, status = 'open' }: { withSession?: boolean; status?: SessionStatus }) {
  const { dispatch } = useAppContext();
  useEffect(() => {
    if (withSession) {
      dispatch({
        type: 'SET_SESSION',
        payload: {
          id: 's1',
          name: 'My Session',
          role: 'edit',
          connectionStatus: 'connected',
          status,
          participants: [{ id: 'p1', name: 'Alice', role: 'edit' }],
        },
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderMenuBar(withSession = true, status: SessionStatus = 'open') {
  return render(
    <AppProvider>
      <Harness withSession={withSession} status={status} />
      <MenuBar />
    </AppProvider>,
  );
}

it('does not show the participant list before any interaction', () => {
  renderMenuBar();
  expect(screen.queryByText('Alice')).not.toBeInTheDocument();
});

it('shows the participant list on hover, without clicking', async () => {
  const user = userEvent.setup();
  renderMenuBar();
  await user.hover(screen.getByText(/人が参加中/));
  expect(screen.getByText('Alice')).toBeInTheDocument();
});

it('hides the hover-revealed list again once the mouse leaves', async () => {
  const user = userEvent.setup();
  renderMenuBar();
  const trigger = screen.getByText(/人が参加中/);
  await user.hover(trigger);
  expect(screen.getByText('Alice')).toBeInTheDocument();
  await user.unhover(trigger);
  expect(screen.queryByText('Alice')).not.toBeInTheDocument();
});

it('still shows the participant list on click (existing behavior preserved)', async () => {
  const user = userEvent.setup();
  renderMenuBar();
  await user.click(screen.getByText(/人が参加中/));
  expect(screen.getByText('Alice')).toBeInTheDocument();
});

describe('online-session menu placement', () => {
  it('with no session: ファイル menu offers join + create, 退出 is present but disabled, and 共同編集 is not shown', async () => {
    const user = userEvent.setup();
    renderMenuBar(false);
    expect(screen.queryByText('共同編集')).not.toBeInTheDocument();
    await user.click(screen.getByText('ファイル'));
    expect(screen.getByText('オンラインセッションに参加')).toBeInTheDocument();
    expect(screen.getByText('オンラインセッションを作成')).toBeInTheDocument();

    const leaveItem = screen.getByText('オンラインセッションを退出');
    await user.click(leaveItem);
    // Disabled — clicking it does nothing observable, no session appears.
    expect(screen.queryByText('共同編集')).not.toBeInTheDocument();
  });

  it('in a session: ファイル menu offers 退出 (join/create disabled), and clicking it leaves the session', async () => {
    const user = userEvent.setup();
    renderMenuBar(true);
    await user.click(screen.getByText('ファイル'));
    expect(screen.getByText('オンラインセッションを退出')).toBeInTheDocument();
    await user.click(screen.getByText('オンラインセッションを退出'));
    expect(screen.queryByText('共同編集')).not.toBeInTheDocument();
  });

  it('in a session: the 共同編集 menu appears with セッション情報 + ロックする (no 終了 item here anymore)', async () => {
    const user = userEvent.setup();
    renderMenuBar(true);
    await user.click(screen.getByText('共同編集'));
    expect(screen.getByText('セッション情報')).toBeInTheDocument();
    expect(screen.getByText('ロックする')).toBeInTheDocument();
    expect(screen.queryByText('セッションを終了')).not.toBeInTheDocument();
  });

  it('clicking ロックする in 共同編集 sends the lock action directly, no dialog', async () => {
    const user = userEvent.setup();
    renderMenuBar(true, 'open');
    await user.click(screen.getByText('共同編集'));
    await user.click(screen.getByText('ロックする'));
    expect(mockedCollab.sendCollabLock).toHaveBeenCalled();
  });

  it('when locked, 共同編集 shows ロック解除 instead, and clicking it sends the unlock action', async () => {
    const user = userEvent.setup();
    renderMenuBar(true, 'lock');
    await user.click(screen.getByText('共同編集'));
    expect(screen.getByText('ロック解除')).toBeInTheDocument();
    await user.click(screen.getByText('ロック解除'));
    expect(mockedCollab.sendCollabUnlock).toHaveBeenCalled();
  });
});
