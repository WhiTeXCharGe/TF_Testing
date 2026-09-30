/**
 * @jest-environment jsdom
 *
 * WorkTaskPanel's start/end date fields used to be read-only display text —
 * this covers the fix that made them editable, dispatching UPDATE_ASSIGNMENT
 * the same way MiscPanel's date fields already did.
 */
import { useEffect } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

let capturedState: ReturnType<typeof useAppContext>['state'] | null = null;

function Harness() {
  const { state, dispatch } = useAppContext();
  capturedState = state;
  useEffect(() => {
    dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
    dispatch({ type: 'SELECT_ASSIGNMENT', payload: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderPanel() {
  return render(<AppProvider><Harness /><SidePanel /></AppProvider>);
}

describe('WorkTaskPanel date editing', () => {
  it('shows the assignment start/end dates as editable date inputs, not read-only text', () => {
    renderPanel();
    const start = screen.getByDisplayValue('2026-01-01') as HTMLInputElement;
    const end = screen.getByDisplayValue('2026-01-05') as HTMLInputElement;
    expect(start.type).toBe('date');
    expect(end.type).toBe('date');
  });

  it('committing a new start date updates the real assignment data', async () => {
    renderPanel();
    const start = screen.getByDisplayValue('2026-01-01') as HTMLInputElement;

    await userEvent.clear(start);
    await userEvent.type(start, '2026-01-02');
    await userEvent.tab();

    await waitFor(() => expect(capturedState?.schedule?.assignmentList[0].startDate).toBe('2026-01-02'));
  });

  it('committing a new end date updates the real assignment data', async () => {
    renderPanel();
    const end = screen.getByDisplayValue('2026-01-05') as HTMLInputElement;

    await userEvent.clear(end);
    await userEvent.type(end, '2026-01-08');
    await userEvent.tab();

    await waitFor(() => expect(capturedState?.schedule?.assignmentList[0].endDate).toBe('2026-01-08'));
  });

  it('does not commit when the new start date would be after the end date', async () => {
    renderPanel();
    const start = screen.getByDisplayValue('2026-01-01') as HTMLInputElement;

    await userEvent.clear(start);
    await userEvent.type(start, '2026-01-09'); // after endDate 2026-01-05
    await userEvent.tab();

    expect(capturedState?.schedule?.assignmentList[0].startDate).toBe('2026-01-01');
  });

  it('the work-hour table still reflects the assignment after its date range changes', async () => {
    renderPanel();
    const end = screen.getByDisplayValue('2026-01-05') as HTMLInputElement;
    await userEvent.clear(end);
    await userEvent.type(end, '2026-01-06');
    await userEvent.tab();

    expect(await screen.findByText(UI.workHourTableTitle)).toBeInTheDocument();
  });
});

// misc_task_list connects to a region directly (unlike workflow_task_list,
// which goes through fab) — MiscPanel didn't show it at all before.
describe('MiscPanel region display', () => {
  const ENV_WITH_REGION: EnvConfig = {
    ...ENV_CONFIG,
    regionList: [{ id: 'r1', name: 'Kansai', unavailableDates: [] }],
  };

  const MISC_SCHEDULE: ScheduleData = {
    planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
    workflowTaskList: [
      { id: 'misc_1', name: 'VISA', workflow: 'wf_misc', region: 'r1', phaseTaskList: [] },
    ],
    assignmentList: [
      {
        worker: 'w1', operationTask: 'misc_1', startDate: '2026-01-01', endDate: '2026-01-05',
        workDateList: [], planFlexibility: 'Fixed', description: '',
      },
    ],
  };

  function MiscHarness() {
    const { dispatch } = useAppContext();
    useEffect(() => {
      dispatch({ type: 'LOAD_FILES', payload: { schedule: MISC_SCHEDULE, envConfig: ENV_WITH_REGION, envPath: 'e.yaml', schedulePath: 's.yaml' } });
      dispatch({ type: 'SELECT_ASSIGNMENT', payload: 0 });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return null;
  }

  it('shows the misc task region by name, resolved from regionList', async () => {
    render(<AppProvider><MiscHarness /><SidePanel /></AppProvider>);
    expect(await screen.findByText('Kansai')).toBeInTheDocument();
  });

  it('shows nothing region-related when the misc task has no region set', async () => {
    const schedule: ScheduleData = {
      ...MISC_SCHEDULE,
      workflowTaskList: [{ id: 'misc_1', name: 'VISA', workflow: 'wf_misc', phaseTaskList: [] }],
    };
    function NoRegionHarness() {
      const { dispatch } = useAppContext();
      useEffect(() => {
        dispatch({ type: 'LOAD_FILES', payload: { schedule, envConfig: ENV_WITH_REGION, envPath: 'e.yaml', schedulePath: 's.yaml' } });
        dispatch({ type: 'SELECT_ASSIGNMENT', payload: 0 });
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return null;
    }
    render(<AppProvider><NoRegionHarness /><SidePanel /></AppProvider>);
    await screen.findByText('VISA'); // panel rendered
    expect(screen.queryByText('Kansai')).not.toBeInTheDocument();
    expect(screen.queryByText(UI.regionFieldLabel)).not.toBeInTheDocument();
  });
});
