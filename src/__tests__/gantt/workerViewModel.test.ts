import { buildWorkerTimelineModel } from '../../components/GanttChart/workerViewModel';
import { EnvConfig } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';

// ── Minimal fixtures ──────────────────────────────────────────────────────────

const ENV: EnvConfig = {
  workflowList: [
    { id: 'wf_std', name: 'Standard', phaseList: [{ id: 'p1', name: 'Setup', operationList: [{ id: 'op1', name: 'Heavy' }] }] },
    { id: 'wf_misc', name: 'Other Work', phaseList: [] },
  ],
  fabList: [{ id: 'fab1', name: 'Osaka', region: 'r1', unavailableDates: [] }],
  regionList: [{ id: 'r1', name: 'Kansai', unavailableDates: [] }],
  customerCompanyList: [],
  workerCompanyList: [{ id: 'co1', name: 'TechCorp', unavailableDates: [] }],
  workerList: [
    { id: 'w001', name: 'Alice', workerCompany: 'co1', unavailableDates: [] },
    { id: 'w002', name: 'Bob',   workerCompany: 'co1', unavailableDates: [] },
    { id: 'w003', name: 'Carol', workerCompany: 'co1', unavailableDates: [{ single: { days: ['2025-09-03'] } }] },
  ],
  transiteDayMap: [],
};

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2025-09-01', endDate: '2025-09-05' },
  workflowTaskList: [
    {
      id: 'wt001', name: 'Module A', workflow: 'wf_std', fab: 'fab1',
      phaseTaskList: [
        {
          id: 'pt1', phase: 'p1', startDate: '2025-09-01', endDate: '2025-09-05',
          operationTaskList: [
            { id: 'ot1', operation: 'op1', workloadHours: 40 },
            { id: 'ot2', operation: 'op1', workloadHours: 40 },
          ],
        },
      ],
    },
  ],
  assignmentList: [
    { worker: 'w001', operationTask: 'ot1', startDate: '2025-09-01', endDate: '2025-09-02', planFlexibility: 'Flexible', workDateList: [] },
    { worker: 'w001', operationTask: 'ot2', startDate: '2025-09-03', endDate: '2025-09-05', planFlexibility: 'Flexible', workDateList: [] },
    { worker: 'w002', operationTask: 'ot1', startDate: '2025-09-01', endDate: '2025-09-05', planFlexibility: 'Flexible', workDateList: [] },
  ],
};

const DATES = ['2025-09-01', '2025-09-02', '2025-09-03', '2025-09-04', '2025-09-05'];

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('buildWorkerTimelineModel', () => {
  let model: ReturnType<typeof buildWorkerTimelineModel>;

  beforeEach(() => {
    model = buildWorkerTimelineModel(ENV, SCHEDULE, DATES, '2025-09-01');
  });

  it('includes workers with assignments', () => {
    const ids = model.rows.map(r => r.workerId);
    expect(ids).toContain('w001');
    expect(ids).toContain('w002');
  });

  it('includes workers with unavailable dates even without assignments', () => {
    const ids = model.rows.map(r => r.workerId);
    expect(ids).toContain('w003');
  });

  it('does not include workers with no assignments and no unavailable dates', () => {
    const envWithExtra: EnvConfig = {
      ...ENV,
      workerList: [...ENV.workerList, { id: 'w_none', name: 'Nobody', unavailableDates: [] }],
    };
    const m = buildWorkerTimelineModel(envWithExtra, SCHEDULE, DATES, '2025-09-01');
    const ids = m.rows.map(r => r.workerId);
    expect(ids).not.toContain('w_none');
  });

  it('fills unavailable day cells for w003', () => {
    const carolRow = model.rows.find(r => r.workerId === 'w003')!;
    expect(carolRow).toBeDefined();
    const sep3Index = DATES.indexOf('2025-09-03');
    expect(carolRow.dayCells[sep3Index].kind).toBe('unavailable');
  });

  it('fills work day cells for w001', () => {
    const aliceRow = model.rows.find(r => r.workerId === 'w001')!;
    const idx = DATES.indexOf('2025-09-01');
    expect(aliceRow.dayCells[idx].kind).toBe('work');
  });

  it('generates correct month groups', () => {
    expect(model.monthGroups).toHaveLength(1);
    expect(model.monthGroups[0].label).toBe('9月');
    expect(model.monthGroups[0].span).toBe(5);
  });

  it('two consecutive same-fab tasks for same worker create separate segments', () => {
    // w001 has ot1 (sep01-02) and ot2 (sep03-05) - different assignmentIndex → separate bars
    const aliceRow = model.rows.find(r => r.workerId === 'w001')!;
    const workSegments = aliceRow.segments.filter(s => s.kind === 'work');
    expect(workSegments.length).toBeGreaterThanOrEqual(2);
  });

  it('worker meta contains company name', () => {
    const aliceRow = model.rows.find(r => r.workerId === 'w001')!;
    expect(aliceRow.meta.company).toBe('TechCorp');
    expect(aliceRow.meta.name).toBe('Alice');
  });

  it('returns only workers with unavailable dates when dates array is empty', () => {
    const m = buildWorkerTimelineModel(ENV, SCHEDULE, [], '2025-09-01');
    // Carol (w003) has unavailable dates so she still appears; w001/w002 have no off-dates
    // and with empty dates no assignments overlap, but they are still in assignedWorkerIds
    // from the schedule — so at least Carol should be present
    const ids = m.rows.map(r => r.workerId);
    expect(ids).toContain('w003');
  });
});

// The unavailable-date red bar reflects each worker's own personal
// unavailableDates only. A company-wide holiday or a region-wide stand-down
// (WorkerCompany.unavailableDates / Region.unavailableDates) was tried too,
// but made real-data timelines noisy, so it's gated off by the
// SHOW_COMPANY_REGION_UNAVAILABLE_DATES constant at the top of
// buildWorkerTimelineModel in workerViewModel.ts — flip that to true (and
// invert the "is unaffected"/"does not affect" assertions below) to turn it
// back on.
describe('buildWorkerTimelineModel — company/region unavailable dates (currently off by design)', () => {
  it('a company-wide unavailable date has no effect — only the worker\'s own unavailableDates count', () => {
    const env: EnvConfig = {
      ...ENV,
      workerCompanyList: [{ id: 'co1', name: 'TechCorp', unavailableDates: [{ single: { days: ['2025-09-02'] } }] }],
    };
    const model = buildWorkerTimelineModel(env, SCHEDULE, DATES, '2025-09-01');
    const idx = DATES.indexOf('2025-09-02');
    const aliceRow = model.rows.find(r => r.workerId === 'w001')!;
    const bobRow = model.rows.find(r => r.workerId === 'w002')!;
    expect(aliceRow.dayCells[idx].kind).toBe('work'); // ot1 runs sep01-02, unaffected by co1's blackout
    expect(bobRow.dayCells[idx].kind).toBe('work');
  });

  it('a company-wide unavailable date does not add a row for a worker with no assignments and no personal unavailable dates', () => {
    const env: EnvConfig = {
      ...ENV,
      workerCompanyList: [{ id: 'co1', name: 'TechCorp', unavailableDates: [{ single: { days: ['2025-09-02'] } }] }],
      workerList: [...ENV.workerList, { id: 'w_idle', name: 'Idle', workerCompany: 'co1', unavailableDates: [] }],
    };
    const model = buildWorkerTimelineModel(env, SCHEDULE, DATES, '2025-09-01');
    expect(model.rows.find(r => r.workerId === 'w_idle')).toBeUndefined();
  });

  it('a region-wide unavailable date has no effect on a worker whose that-day task is in that region', () => {
    const env: EnvConfig = {
      ...ENV,
      regionList: [{ id: 'r1', name: 'Kansai', unavailableDates: [{ single: { days: ['2025-09-04'] } }] }],
    };
    const model = buildWorkerTimelineModel(env, SCHEDULE, DATES, '2025-09-01');
    const idx = DATES.indexOf('2025-09-04');
    // w001's ot2 (sep03-05) is fab1 → region r1
    const aliceRow = model.rows.find(r => r.workerId === 'w001')!;
    expect(aliceRow.dayCells[idx].kind).toBe('work');
  });
});
