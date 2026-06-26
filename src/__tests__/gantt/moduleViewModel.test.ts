import { buildModuleViewModel } from '../../components/GanttChart/moduleViewModel';
import { EnvConfig } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';

// ── Fixtures ──────────────────────────────────────────────────────────────────

const ENV: EnvConfig = {
  workflowList: [
    {
      id: 'wf_std', name: 'Standard', phaseList: [
        { id: 'p1', name: 'Module Setup', operationList: [
          { id: 'op1', name: 'Heavy', workHours: [8], workloadHours: 160, minWorkerNum: 2, maxWorkerNum: 3 },
          { id: 'op2', name: 'Elec',  workHours: [8], workloadHours: 80,  minWorkerNum: 1, maxWorkerNum: 2 },
        ]},
        { id: 'p2', name: 'Function Setup', operationList: [
          { id: 'op3', name: 'Test', workHours: [8], workloadHours: 40, minWorkerNum: 1, maxWorkerNum: 1 },
        ]},
      ],
    },
    { id: 'wf_misc', name: 'Other Work', phaseList: [] },
  ],
  fabList: [{ id: 'fab1', name: 'Osaka', region: 'r1', unavailableDates: [] }],
  regionList: [{ id: 'r1', name: 'Kansai', unavailableDates: [] }],
  customerCompanyList: [],
  workerCompanyList: [{ id: 'co1', name: 'TechCorp', unavailableDates: [] }],
  workerList: [
    { id: 'w001', name: 'Alice', workerCompany: 'co1', unavailableDates: [] },
    { id: 'w002', name: 'Bob',   workerCompany: 'co1', unavailableDates: [] },
  ],
  transiteDayMap: [],
};

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2025-09-01', endDate: '2025-09-30' },
  workflowTaskList: [
    {
      id: 'wt001', name: 'SU-1001', workflow: 'wf_std', fab: 'fab1',
      phaseTaskList: [
        {
          id: 'pt1', phase: 'p1', startDate: '2025-09-01', endDate: '2025-09-15',
          operationTaskList: [
            { id: 'ot1', operation: 'op1', workloadHours: 160 },
            { id: 'ot2', operation: 'op2', workloadHours: 80  },
          ],
        },
        {
          id: 'pt2', phase: 'p2', startDate: '2025-09-16', endDate: '2025-09-30',
          operationTaskList: [
            { id: 'ot3', operation: 'op3', workloadHours: 40 },
          ],
        },
      ],
    },
    // misc task — should be FILTERED OUT
    { id: 'wt_misc', name: 'Other Work Job', workflow: 'wf_misc', phaseTaskList: [] },
  ],
  assignmentList: [
    { worker: 'w001', operationTask: 'ot1', startDate: '2025-09-01', endDate: '2025-09-10', planFlexibility: 'Flexible', workDateList: [] },
    { worker: 'w002', operationTask: 'ot2', startDate: '2025-09-03', endDate: '2025-09-12', planFlexibility: 'Flexible', workDateList: [] },
    { worker: 'w001', operationTask: 'ot3', startDate: '2025-09-16', endDate: '2025-09-25', planFlexibility: 'Flexible', workDateList: [] },
  ],
};

const DATES: string[] = (() => {
  const d: string[] = [];
  let cur = '2025-09-01';
  while (cur <= '2025-09-30') {
    d.push(cur);
    const dt = new Date(`${cur}T00:00:00`);
    dt.setDate(dt.getDate() + 1);
    cur = dt.toISOString().slice(0, 10);
  }
  return d;
})();

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('buildModuleViewModel', () => {
  let model: ReturnType<typeof buildModuleViewModel>;

  beforeEach(() => {
    model = buildModuleViewModel(ENV, SCHEDULE, DATES);
  });

  it('excludes misc tasks (empty phaseTaskList)', () => {
    const ids = model.modules.map(m => m.moduleId);
    expect(ids).not.toContain('wt_misc');
    expect(ids).toContain('wt001');
  });

  it('contains correct module count', () => {
    expect(model.modules).toHaveLength(1);
  });

  it('sets moduleName from workflowTask.name', () => {
    expect(model.modules[0].moduleName).toBe('SU-1001');
  });

  it('exposes fab and region on ModuleNode', () => {
    expect(model.modules[0].fab).toBe('fab1');
  });

  it('builds correct phase count', () => {
    expect(model.modules[0].phases).toHaveLength(2);
  });

  it('phase names are set from EnvConfig', () => {
    const names = model.modules[0].phases.map(p => p.phaseName);
    expect(names).toContain('Module Setup');
    expect(names).toContain('Function Setup');
  });

  it('phase barStartDate equals earliest assignment start', () => {
    const phase1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    expect(phase1.barStartDate).toBe('2025-09-01');
  });

  it('phase barEndDate equals latest assignment end', () => {
    const phase1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    expect(phase1.barEndDate).toBe('2025-09-12');
  });

  it('tasks have correct workloadHours', () => {
    const pt1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    const ot1 = pt1.tasks.find(t => t.taskId === 'ot1')!;
    expect(ot1.workloadHours).toBe(160);
  });

  it('tasks use envConfig defaults for minWorker when not in schedule', () => {
    const pt1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    const ot1 = pt1.tasks.find(t => t.taskId === 'ot1')!;
    // ot1 has no recommendsWorkerMin in schedule → falls back to op.minWorkerNum = 2
    expect(ot1.minWorker).toBe(2);
  });

  it('workerCount per phase counts distinct workers', () => {
    const phase1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    // w001 on ot1, w002 on ot2 → 2 workers
    expect(phase1.workerCount).toBe(2);
  });

  it('slot list contains assignment worker IDs', () => {
    const pt1 = model.modules[0].phases.find(p => p.phaseId === 'pt1')!;
    const ot1 = pt1.tasks.find(t => t.taskId === 'ot1')!;
    expect(ot1.slots.map(s => s.workerId)).toContain('w001');
  });

  it('generates month groups', () => {
    expect(model.monthGroups).toHaveLength(1);
    expect(model.monthGroups[0].label).toBe('9月');
  });

  it('returns empty modules for empty schedule', () => {
    const emptySchedule: ScheduleData = { planRange: SCHEDULE.planRange, workflowTaskList: [], assignmentList: [] };
    const m = buildModuleViewModel(ENV, emptySchedule, DATES);
    expect(m.modules).toHaveLength(0);
  });
});
