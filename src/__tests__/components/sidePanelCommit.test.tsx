/**
 * @jest-environment jsdom
 *
 * SidePanel fields commit on blur. Clicking elsewhere in the app (another bar,
 * the background) deselects and unmounts/re-targets the panel on mousedown,
 * before the browser moves focus — so a pending edit used to be dropped.
 */
import { useEffect } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { SidePanel } from '../../components/SidePanel/SidePanel';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { UI } from '../../config/uiText';

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
            { id: 'ot2', operation: 'op2', workloadHours: 10, colorCode: '00FF00' },
          ],
        },
      ],
    },
  ],
  assignmentList: [
    { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-01', endDate: '2026-01-05', workDateList: [{ date: '2026-01-01', hour: 8 }], planFlexibility: 'Flexible', description: '' },
    { worker: 'w1', operationTask: 'ot2', startDate: '2026-01-06', endDate: '2026-01-08', workDateList: [{ date: '2026-01-06', hour: 8 }], planFlexibility: 'Flexible', description: '' },
  ],
};

let latest: ReturnType<typeof useAppContext> | null = null;

function Harness() {
  const ctx = useAppContext();
  latest = ctx;
  const { state, dispatch } = ctx;
  useEffect(() => {
    dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
    dispatch({ type: 'SELECT_ASSIGNMENT', payload: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!state.schedule) return null;
  return (
    <>
      <div data-testid="outside">outside</div>
      <SidePanel />
    </>
  );
}

function setup() {
  render(<AppProvider><Harness /></AppProvider>);
}

describe('SidePanel commits pending edits when the user clicks elsewhere', () => {
  it('saves the remarks text when the mousedown lands outside the panel', () => {
    setup();
    const remarks = screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement;
    act(() => { remarks.focus(); });
    fireEvent.change(remarks, { target: { value: 'typed note' } });

    fireEvent.mouseDown(screen.getByTestId('outside'));

    expect(latest!.state.schedule!.assignmentList[0].description).toBe('typed note');
    expect(latest!.state.selectedAssignmentIndex).toBeNull();
  });

  it('saves a changed date when the mousedown lands outside the panel', () => {
    const { container } = render(<AppProvider><Harness /></AppProvider>);
    const [startInput] = Array.from(container.querySelectorAll('input[type="date"]')) as HTMLInputElement[];
    act(() => { startInput.focus(); });
    fireEvent.change(startInput, { target: { value: '2026-01-02' } });

    fireEvent.mouseDown(screen.getByTestId('outside'));

    expect(latest!.state.schedule!.assignmentList[0].startDate).toBe('2026-01-02');
  });

  it('applies the edit to the assignment it was made on, not the one clicked next', () => {
    setup();
    const remarks = screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement;
    act(() => { remarks.focus(); });
    fireEvent.change(remarks, { target: { value: 'for first' } });

    // Mimics clicking another bar: mousedown first, then the bar's selection.
    const outside = screen.getByTestId('outside');
    fireEvent.mouseDown(outside);
    act(() => { latest!.dispatch({ type: 'SELECT_ASSIGNMENT', payload: 1 }); });

    expect(latest!.state.schedule!.assignmentList[0].description).toBe('for first');
    expect(latest!.state.schedule!.assignmentList[1].description).toBe('');
    expect((screen.getByPlaceholderText(UI.remarksPlaceholder) as HTMLTextAreaElement).value).toBe('');
  });
});
