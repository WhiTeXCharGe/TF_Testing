/**
 * @jest-environment jsdom
 */
import { useEffect } from 'react';
import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { ModuleFlexibilityDialog } from '../../components/Toolbar/ModuleFlexibilityDialog';
import { Toolbar } from '../../components/Toolbar/Toolbar';
import { ScheduleData, Assignment, PlanFlexibility } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';
import { UI } from '../../config/uiText';

const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [],
  workerList: [{ id: 'w1', name: 'Worker One', unavailableDates: [] }],
  transiteDayMap: [],
};

const op = (id: string) => ({ id, operation: id, workloadHours: 8, colorCode: 'FF0000' });
const asg = (task: string): Assignment => ({
  worker: 'w1', operationTask: task, startDate: '2026-01-01', endDate: '2026-01-02',
  workDateList: [], planFlexibility: 'Flexible', description: '',
});

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [
    {
      id: 'wtA', name: 'ALPHA-100', workflow: 'wf',
      phaseTaskList: [
        { id: 'pA1', name: 'PhaseA1', phase: 'p', startDate: '2026-01-01', endDate: '2026-01-10', operationTaskList: [op('oA1'), op('oA2')] },
        { id: 'pA2', name: 'PhaseA2', phase: 'p', startDate: '2026-01-01', endDate: '2026-01-10', operationTaskList: [op('oA3')] },
      ],
    },
    {
      id: 'wtB', name: 'BETA-200', workflow: 'wf',
      phaseTaskList: [
        { id: 'pB1', name: 'PhaseB1', phase: 'p', startDate: '2026-01-01', endDate: '2026-01-10', operationTaskList: [op('oB1')] },
      ],
    },
  ],
  assignmentList: [asg('oA1'), asg('oA2'), asg('oA3'), asg('oB1')],
};

let api: ReturnType<typeof useAppContext>;
function Harness() {
  api = useAppContext();
  const { state, dispatch } = api;
  useEffect(() => {
    dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!state.schedule) return null;
  return <ModuleFlexibilityDialog />;
}

const flexOf = () => api.state.schedule!.assignmentList.map(a => a.planFlexibility as PlanFlexibility);

function openDialog() {
  render(<AppProvider><Harness /></AppProvider>);
  fireEvent.click(screen.getByRole('button', { name: UI.moduleFlexBtn }));
}
const addModule = (name: string) => fireEvent.click(screen.getByText(`+ ${name}`));
const pick = (label: string, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
const apply = () => fireEvent.click(screen.getByRole('button', { name: UI.bulkApply }));

describe('ModuleFlexibilityDialog', () => {
  it('filters the module list by the search text', () => {
    openDialog();
    expect(screen.getByText('+ ALPHA-100')).toBeInTheDocument();
    expect(screen.getByText('+ BETA-200')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText(UI.moduleFlexSearchPlaceholder), { target: { value: 'beta' } });
    expect(screen.queryByText('+ ALPHA-100')).not.toBeInTheDocument();
    expect(screen.getByText('+ BETA-200')).toBeInTheDocument();
  });

  it('module-level choice sets every 工程 and 作業 of that module only', () => {
    openDialog();
    addModule('ALPHA-100');
    pick(`ALPHA-100 ${UI.moduleFlexModuleLevel}`, 'Fixed');
    apply();
    expect(flexOf()).toEqual(['Fixed', 'Fixed', 'Fixed', 'Flexible']);
  });

  it('phase-level choice only affects that phase, and 作業-level only that task', () => {
    openDialog();
    addModule('ALPHA-100');
    fireEvent.click(screen.getByRole('button', { name: UI.moduleFlexExpand }));
    pick('ALPHA-100 PhaseA1', 'Reluctant');
    // expand phase A1 to reach its 作業 selects
    fireEvent.click(within(screen.getByTestId('flex-module-wtA')).getAllByRole('button', { name: UI.moduleFlexExpand })[0]);
    pick('ALPHA-100 PhaseA1 oA2', 'Fixed');
    apply();
    expect(flexOf()).toEqual(['Reluctant', 'Fixed', 'Flexible', 'Flexible']);
  });

  it('shows 混在 at module level when its parts differ', () => {
    openDialog();
    addModule('ALPHA-100');
    fireEvent.click(screen.getByRole('button', { name: UI.moduleFlexExpand }));
    pick('ALPHA-100 PhaseA2', 'Fixed');
    const moduleSelect = screen.getByLabelText(`ALPHA-100 ${UI.moduleFlexModuleLevel}`) as HTMLSelectElement;
    expect(within(moduleSelect).getByText(UI.moduleFlexMixed)).toBeInTheDocument();
  });

  it('the one-shot selector sets every added module together', () => {
    openDialog();
    addModule('ALPHA-100');
    addModule('BETA-200');
    pick(UI.moduleFlexSetAllLabel, 'Fixed');
    apply();
    expect(flexOf()).toEqual(['Fixed', 'Fixed', 'Fixed', 'Fixed']);
  });

  it('a removed module is not changed even if it was edited first', () => {
    openDialog();
    addModule('ALPHA-100');
    addModule('BETA-200');
    pick(`BETA-200 ${UI.moduleFlexModuleLevel}`, 'Fixed');
    fireEvent.click(within(screen.getByTestId('flex-module-wtB')).getByRole('button', { name: UI.moduleFlexRemove }));
    pick(`ALPHA-100 ${UI.moduleFlexModuleLevel}`, 'Reluctant');
    apply();
    expect(flexOf()).toEqual(['Reluctant', 'Reluctant', 'Reluctant', 'Flexible']);
  });

  it('can be undone in one step', () => {
    openDialog();
    addModule('BETA-200');
    pick(UI.moduleFlexSetAllLabel, 'Fixed');
    apply();
    expect(flexOf()).toEqual(['Flexible', 'Flexible', 'Flexible', 'Fixed']);
    act(() => api.dispatch({ type: 'UNDO' }));
    expect(flexOf()).toEqual(['Flexible', 'Flexible', 'Flexible', 'Flexible']);
  });

  it('OK stays disabled until something is changed', () => {
    openDialog();
    addModule('ALPHA-100');
    expect(screen.getByRole('button', { name: UI.bulkApply })).toBeDisabled();
  });
});

describe('Toolbar placement', () => {
  function ToolbarHarness({ view }: { view: 'device' | 'worker' }) {
    const { state, dispatch } = useAppContext();
    useEffect(() => {
      dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } });
      dispatch({ type: 'SWITCH_VIEW', payload: view });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    if (!state.schedule) return null;
    return <Toolbar />;
  }

  it('offers the button in the module (device) view', () => {
    render(<AppProvider><ToolbarHarness view="device" /></AppProvider>);
    expect(screen.getByRole('button', { name: UI.moduleFlexBtn })).toBeInTheDocument();
  });

  it('does not offer it in the worker view', () => {
    render(<AppProvider><ToolbarHarness view="worker" /></AppProvider>);
    expect(screen.queryByRole('button', { name: UI.moduleFlexBtn })).not.toBeInTheDocument();
  });
});
