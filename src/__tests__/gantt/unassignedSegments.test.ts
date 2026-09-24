// Kept separate from moduleViewModel.test.ts, which is excluded from the
// suite (jest.config.cjs testPathIgnorePatterns) because its buildModuleViewModel
// fixtures crash the jest worker with an out-of-memory error — pre-existing,
// unrelated to this helper. unassignedSegments doesn't touch
// buildModuleViewModel at all, so it's safe to test on its own.
import { unassignedSegments, ModulePhase } from '../../components/GanttChart/moduleViewModel';

const phase = (over: Partial<ModulePhase>): ModulePhase => ({
  moduleId: 'm1', phaseId: 'p', phaseName: 'P', planStartDate: '2025-09-01', planEndDate: '2025-09-10',
  barStartDate: null, barEndDate: null, workerCount: 0, tasks: [], color: '#fff', ...over,
});

describe('unassignedSegments', () => {
  it('returns nothing when every phase already has a worker assigned', () => {
    const phases = [phase({ phaseId: 'a', workerCount: 1 }), phase({ phaseId: 'b', workerCount: 2 })];
    expect(unassignedSegments(phases)).toEqual([]);
  });

  it('returns nothing for an empty phase list', () => {
    expect(unassignedSegments([])).toEqual([]);
  });

  it("a single unassigned phase with no neighbors spans its own planned range", () => {
    const phases = [phase({ phaseId: 'a', planStartDate: '2025-09-05', planEndDate: '2025-09-20', workerCount: 0 })];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-05', end: '2025-09-20' }]);
  });

  it('every phase unassigned collapses into one segment spanning first-to-last plan dates', () => {
    const phases = [
      phase({ phaseId: 'a', planStartDate: '2025-09-01', planEndDate: '2025-09-05', workerCount: 0 }),
      phase({ phaseId: 'b', planStartDate: '2025-09-06', planEndDate: '2025-09-12', workerCount: 0 }),
      phase({ phaseId: 'c', planStartDate: '2025-09-13', planEndDate: '2025-09-30', workerCount: 0 }),
    ];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-01', end: '2025-09-30' }]);
  });

  it('a gap between two assigned phases is bounded by their actual bar dates, not overlapping either', () => {
    const phases = [
      phase({ phaseId: '1', workerCount: 2, barStartDate: '2025-09-01', barEndDate: '2025-09-05' }),
      phase({ phaseId: '2', planStartDate: '2025-09-06', planEndDate: '2025-09-15', workerCount: 0 }),
      phase({ phaseId: '3', workerCount: 1, barStartDate: '2025-09-16', barEndDate: '2025-09-20' }),
    ];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-06', end: '2025-09-15' }]);
  });

  it('a leading gap falls back to the module start date on its open side', () => {
    const phases = [
      phase({ phaseId: '1', planStartDate: '2025-09-01', planEndDate: '2025-09-04', workerCount: 0 }),
      phase({ phaseId: '2', workerCount: 1, barStartDate: '2025-09-10', barEndDate: '2025-09-15' }),
    ];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-01', end: '2025-09-09' }]);
  });

  it('a trailing gap falls back to the module end date on its open side', () => {
    const phases = [
      phase({ phaseId: '1', workerCount: 1, barStartDate: '2025-09-01', barEndDate: '2025-09-05' }),
      phase({ phaseId: '2', planStartDate: '2025-09-06', planEndDate: '2025-09-20', workerCount: 0 }),
    ];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-06', end: '2025-09-20' }]);
  });

  it('produces one segment per gap when scheduled and unscheduled phases alternate (1,2,3,4 with 1 and 3 scheduled)', () => {
    const phases = [
      phase({ phaseId: '1', workerCount: 1, barStartDate: '2025-09-01', barEndDate: '2025-09-05' }),
      phase({ phaseId: '2', planStartDate: '2025-09-06', planEndDate: '2025-09-09', workerCount: 0 }),
      phase({ phaseId: '3', workerCount: 1, barStartDate: '2025-09-10', barEndDate: '2025-09-14' }),
      phase({ phaseId: '4', planStartDate: '2025-09-15', planEndDate: '2025-09-25', workerCount: 0 }),
    ];
    expect(unassignedSegments(phases)).toEqual([
      { start: '2025-09-06', end: '2025-09-09' },
      { start: '2025-09-15', end: '2025-09-25' },
    ]);
  });

  it('skips a gap with no room between back-to-back assigned phases', () => {
    const phases = [
      phase({ phaseId: '1', workerCount: 1, barStartDate: '2025-09-01', barEndDate: '2025-09-05' }),
      phase({ phaseId: '2', planStartDate: '2025-09-06', planEndDate: '2025-09-06', workerCount: 0 }),
      phase({ phaseId: '3', workerCount: 1, barStartDate: '2025-09-06', barEndDate: '2025-09-10' }),
    ];
    expect(unassignedSegments(phases)).toEqual([]);
  });

  it('falls back to plan dates for an assigned neighbor with no bar dates yet', () => {
    const phases = [
      phase({ phaseId: '1', planStartDate: '2025-09-01', planEndDate: '2025-09-05', workerCount: 1, barStartDate: null, barEndDate: null }),
      phase({ phaseId: '2', planStartDate: '2025-09-06', planEndDate: '2025-09-15', workerCount: 0 }),
    ];
    expect(unassignedSegments(phases)).toEqual([{ start: '2025-09-06', end: '2025-09-15' }]);
  });
});
