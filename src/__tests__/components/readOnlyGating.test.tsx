/**
 * @jest-environment jsdom
 *
 * Covers the read-only (view-role) gating added across Toolbar,
 * UndoRedoButtons, and SidePanel: a view-role participant's editable
 * controls must be disabled/inert, while the exact same controls stay
 * enabled for an edit-role session and for solo mode (no session at all).
 */
import { useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { Toolbar } from '../../components/Toolbar/Toolbar';
import { UndoRedoButtons } from '../../components/Toolbar/UndoRedoButtons';
import { SidePanel } from '../../components/SidePanel/SidePanel';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { UI } from '../../config/uiText';
import { SessionRole } from '../../types/appState';

const ENV_CONFIG: EnvConfig = {
  workflowList: [],
  fabList: [],
  regionList: [],
  customerCompanyList: [],
  workerCompanyList: [],
  workerList: [{ id: 'w1', name: 'Worker One', unavailableDates: [] }],
  transiteDayMap: [],
};

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [
    {
      id: 'wt1',
      workflow: 'wf1',
      phaseTaskList: [
        {
          id: 'pt1',
          phase: 'ph1',
          startDate: '2026-01-01',
          endDate: '2026-01-10',
          operationTaskList: [
            { id: 'ot1', operation: 'op1', workloadHours: 10, colorCode: 'FF0000' },
          ],
        },
      ],
    },
  ],
  assignmentList: [
    {
      worker: 'w1',
      operationTask: 'ot1',
      startDate: '2026-01-01',
      endDate: '2026-01-05',
      workDateList: [{ date: '2026-01-01', hour: 8 }],
      planFlexibility: 'Flexible',
      description: '',
    },
  ],
};

// Loads a schedule (making one edit first, so the undo stack is non-empty),
// selects the sole assignment (so SidePanel's WorkTaskPanel renders), then
// optionally attaches a session with the given role. SET_SESSION (unlike
// SET_SESSION_BASELINE) doesn't clear the undo stack, so the prior edit
// survives — letting the undo/redo test assert on role alone, not on
// whether there's anything to undo.
function Harness({ role, children }: { role?: SessionRole; children: React.ReactNode }) {
  const { state, dispatch } = useAppContext();
  useEffect(() => {
    dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
    dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { description: 'edited' } } });
    dispatch({ type: 'SELECT_ASSIGNMENT', payload: 0 });
    if (role) {
      dispatch({ type: 'SET_SESSION', payload: { id: 's1', name: 'Test Session', role, connectionStatus: 'connected', participants: [] } });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!state.schedule) return null;
  return <>{children}</>;
}

function renderWithRole(role: SessionRole | undefined, ui: React.ReactNode) {
  return render(
    <AppProvider>
      <Harness role={role}>{ui}</Harness>
    </AppProvider>,
  );
}

describe('Toolbar read-only gating', () => {
  it('disables the add-bar button for a view-role session', () => {
    renderWithRole('view', <Toolbar />);
    expect(screen.getByRole('button', { name: UI.addBarBtn })).toBeDisabled();
  });

  it('keeps the add-bar button enabled for an edit-role session', () => {
    renderWithRole('edit', <Toolbar />);
    expect(screen.getByRole('button', { name: UI.addBarBtn })).toBeEnabled();
  });

  it('keeps the add-bar button enabled in solo mode (no session)', () => {
    renderWithRole(undefined, <Toolbar />);
    expect(screen.getByRole('button', { name: UI.addBarBtn })).toBeEnabled();
  });
});

describe('UndoRedoButtons read-only gating', () => {
  it('disables undo/redo for a view-role session even with a non-empty undo stack', () => {
    renderWithRole('view', <UndoRedoButtons />);
    expect(screen.getByRole('button', { name: UI.undo })).toBeDisabled();
    expect(screen.getByRole('button', { name: UI.redo })).toBeDisabled();
  });

  it('keeps undo enabled for an edit-role session with a non-empty undo stack', () => {
    renderWithRole('edit', <UndoRedoButtons />);
    expect(screen.getByRole('button', { name: UI.undo })).toBeEnabled();
  });

  it('keeps undo enabled in solo mode with a non-empty undo stack', () => {
    renderWithRole(undefined, <UndoRedoButtons />);
    expect(screen.getByRole('button', { name: UI.undo })).toBeEnabled();
  });
});

describe('SidePanel read-only gating (WorkTaskPanel)', () => {
  it('disables the delete button, flexibility select, color picker, remarks textarea, and work-hour input for a view-role session', () => {
    const { container } = renderWithRole('view', <SidePanel />);

    expect(screen.getByRole('button', { name: UI.deleteButton })).toBeDisabled();
    expect(screen.getByDisplayValue(UI.flexibleDesc)).toBeDisabled();

    const colorInput = container.querySelector('input[type="color"]') as HTMLInputElement;
    expect(colorInput).toBeDisabled();

    const remarks = screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement;
    expect(remarks).toHaveAttribute('readonly');

    const hourInput = screen.getByDisplayValue('8') as HTMLInputElement;
    expect(hourInput).toHaveAttribute('readonly');
  });

  it('keeps the same controls enabled/editable for an edit-role session', () => {
    const { container } = renderWithRole('edit', <SidePanel />);

    expect(screen.getByRole('button', { name: UI.deleteButton })).toBeEnabled();
    expect(screen.getByDisplayValue(UI.flexibleDesc)).toBeEnabled();

    const colorInput = container.querySelector('input[type="color"]') as HTMLInputElement;
    expect(colorInput).toBeEnabled();

    const remarks = screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement;
    expect(remarks).not.toHaveAttribute('readonly');

    const hourInput = screen.getByDisplayValue('8') as HTMLInputElement;
    expect(hourInput).not.toHaveAttribute('readonly');
  });

  it('keeps the same controls enabled/editable in solo mode (no session)', () => {
    const { container } = renderWithRole(undefined, <SidePanel />);

    expect(screen.getByRole('button', { name: UI.deleteButton })).toBeEnabled();
    expect(screen.getByDisplayValue(UI.flexibleDesc)).toBeEnabled();

    const colorInput = container.querySelector('input[type="color"]') as HTMLInputElement;
    expect(colorInput).toBeEnabled();

    const remarks = screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement;
    expect(remarks).not.toHaveAttribute('readonly');

    const hourInput = screen.getByDisplayValue('8') as HTMLInputElement;
    expect(hourInput).not.toHaveAttribute('readonly');
  });
});
