/**
 * @jest-environment jsdom
 *
 * An unavailable-date bar must look the same whether or not another bar is
 * being dragged. It used to be faded while idle (an accident of
 * `undefined === undefined` in the in-drag check) and snap to full red the
 * moment any other bar started moving.
 */
import { useEffect } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { WorkerViewGantt } from '../../components/GanttChart/WorkerViewGantt';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { generateDateRange } from '../../utils/dateUtils';

const ENV_CONFIG = {
  workflowList: [],
  fabList: [],
  regionList: [],
  customerCompanyList: [],
  workerCompanyList: [],
  workerList: [{ id: 'w1', name: 'Worker One', unavailableDates: [{ date: '2026-01-10' }] }],
  transiteDayMap: [],
} as unknown as EnvConfig;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-20' },
  workflowTaskList: [
    {
      id: 'wt1',
      workflow: 'wf1',
      phaseTaskList: [
        {
          id: 'pt1', phase: 'ph1', startDate: '2026-01-01', endDate: '2026-01-20',
          operationTaskList: [{ id: 'ot1', operation: 'op1', workloadHours: 10, colorCode: 'FF0000' }],
        },
      ],
    },
  ],
  assignmentList: [
    { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-02', endDate: '2026-01-05', workDateList: [{ date: '2026-01-02', hour: 8 }], planFlexibility: 'Flexible', description: '' },
  ],
};

const DATES = generateDateRange('2026-01-01', '2026-01-20');

function Harness() {
  const { state, dispatch } = useAppContext();
  useEffect(() => {
    dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!state.schedule) return null;
  return <WorkerViewGantt dates={DATES} />;
}

describe('WorkerViewGantt unavailable bar opacity', () => {
  it('is unchanged while another bar is being dragged', () => {
    render(<AppProvider><Harness /></AppProvider>);
    const unavailable = screen.getByTestId('unavailable-bar');
    const idle = unavailable.style.opacity;

    fireEvent.mouseDown(screen.getByTestId('assignment-bar'), { clientX: 100, clientY: 20 });
    act(() => { fireEvent.mouseMove(window, { clientX: 160, clientY: 20 }); });

    expect(screen.getByTestId('unavailable-bar').style.opacity).toBe(idle);
    act(() => { fireEvent.mouseUp(window); });
    expect(screen.getByTestId('unavailable-bar').style.opacity).toBe(idle);
  });
});
