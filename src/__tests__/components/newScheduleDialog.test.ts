import { resolvePhaseEndDates, buildWorkflowTask, WTEntry } from '../../components/Dialogs/NewScheduleDialog';
import { Workflow } from '../../types/envConfig';

describe('resolvePhaseEndDates', () => {
  it('keeps every entered end date as-is', () => {
    const out = resolvePhaseEndDates(['2026-01-16', '2026-01-30', '2026-03-01'], 3, '2025-12-19');
    expect(out).toEqual(['2026-01-16', '2026-01-30', '2026-03-01']);
  });

  it('fills a blank earlier phase with the next phase end date', () => {
    const out = resolvePhaseEndDates(['', '', '2026-03-01'], 3, '2025-12-19');
    expect(out).toEqual(['2026-03-01', '2026-03-01', '2026-03-01']);
  });

  it('fills only the blank phases, propagating from the nearest later filled one', () => {
    const out = resolvePhaseEndDates(['', '2026-01-30', ''], 3, '2025-12-19');
    // pi=2 blank -> fallback (start); pi=1 filled; pi=0 blank -> inherits pi=1
    expect(out).toEqual(['2026-01-30', '2026-01-30', '2025-12-19']);
  });

  it('trims whitespace-only values and treats them as blank', () => {
    const out = resolvePhaseEndDates(['  ', '2026-02-01'], 2, '2025-12-19');
    expect(out).toEqual(['2026-02-01', '2026-02-01']);
  });

  it('falls back to the start date when nothing downstream is filled', () => {
    const out = resolvePhaseEndDates(['', '', ''], 3, '2025-12-19');
    expect(out).toEqual(['2025-12-19', '2025-12-19', '2025-12-19']);
  });
});

describe('buildWorkflowTask', () => {
  const workflow: Workflow = {
    id: 'wf_tool',
    name: 'Tool',
    phaseList: [
      { id: 'p2', name: 'Hardware Setup', operationList: [{ id: 'p2o1', name: 'Mech' }] },
      { id: 'p3', name: 'Function Setup', operationList: [{ id: 'p3o1', name: 'QC' }] },
      { id: 'p4', name: 'Acceptance Inspection', operationList: [{ id: 'p4o1', name: 'QC' }] },
    ],
  };

  const makeEntry = (phaseEndDates: string[]): WTEntry => ({
    key: 'e1',
    name: 'Micron 830300296A',
    workflowId: 'wf_tool',
    fabId: 'f17',
    startDate: '2026-12-19',
    phaseEndDates,
    collapsed: false,
    opEntries: workflow.phaseList.map(ph => ph.operationList.map(() => ({ workloadHours: 100, minWorker: 1, maxWorker: 2 }))),
  });

  it('gives every phase a non-empty window when only the last end date is entered', () => {
    const wt = buildWorkflowTask(makeEntry(['', '', '2027-03-01']), workflow);
    expect(wt.phaseTaskList.map(pt => [pt.startDate, pt.endDate])).toEqual([
      ['2026-12-19', '2027-03-01'],
      ['2026-12-19', '2027-03-01'],
      ['2026-12-19', '2027-03-01'],
    ]);
  });

  it('respects per-phase end dates when all are entered', () => {
    const wt = buildWorkflowTask(makeEntry(['2027-01-16', '2027-01-30', '2027-03-01']), workflow);
    expect(wt.phaseTaskList.map(pt => pt.endDate)).toEqual(['2027-01-16', '2027-01-30', '2027-03-01']);
  });

  it('generates stable phase and operation task ids', () => {
    const wt = buildWorkflowTask(makeEntry(['', '', '2027-03-01']), workflow);
    expect(wt.phaseTaskList.map(pt => pt.id)).toEqual([
      `${wt.id}_p0`, `${wt.id}_p1`, `${wt.id}_p2`,
    ]);
    expect(wt.phaseTaskList[0].operationTaskList[0].id).toBe(`${wt.id}_p0_o0`);
  });
});
